import cv2
import numpy as np
import os
import time
from PIL import Image


REFINE_RADIUS_XY = 1
REFINE_RADIUS_THETA = 0.25
REFINE_THETA_STEP = 0.25
ROI_FRAC = 0.6
MIN_OVERLAP = 20
STRIP_WIDTH = 50

#---- core allignment functions ----#
def preprocess(frame, scale=1.0) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = cv2.resize(gray, (int(w*scale), int(h*scale)))
    return gray

def choose_search_ranges(gray_shape: tuple[int, int],
                        min_size: int = 80)-> tuple[int, int, float, float, int]:
    """
    Choose motion estimation parameters based on the input video properties.

    Returns:
        max_shift_x (int): max |dx| in pixels (after preprocess) of level 0
        max_shift_y (int): max |dy| in pixels (after preprocess) of level 0
        theta_max (float): max rotation in degrees
        theta_step (float): rotation step in degrees
        num_levels (int): number of pyramid levels
    """
    H, W = gray_shape

    # translation range
    max_shift_x = int(round(0.10 * W))
    max_shift_x = int(np.clip(max_shift_x, 6, 40))

    max_shift_y = int(round(0.02 * H))             
    max_shift_y = int(np.clip(max_shift_y, 1, 6))

    # rotation range
    theta_max = 1.0                            
    theta_step = 0.25


    # number of pyramid levels
    
    ratio = min(H, W) / float(min_size)
    max_levels_by_size = int(np.floor(np.log2(ratio))) if ratio >= 1.0 else 0
    
    num_levels = 1 + min(max_levels_by_size, 2)      # at most 3 levels total
    num_levels = int(np.clip(num_levels, 1, 3))

    return max_shift_x, max_shift_y, theta_max, theta_step, num_levels

def estimate_rigid(
    g0: np.ndarray,
    g1: np.ndarray,
    max_shift_x: int,
    max_shift_y: int,
    theta_max: float,
    theta_step: float,
    roi_frac: float = 0.6
) -> tuple[float, int, int]:
    """
    Estimate a small rigid motion between consecutive grayscale frames by discrete search:
    theta in [-theta_max, +theta_max] with step theta_step,
    and (dx, dy) in the provided integer ranges.

    Returns:
        best_theta_deg (float), best_dx (int), best_dy (int)
    """

    H, W = g0.shape
    g0f = g0.astype(np.float32, copy=False)
    g1f = g1.astype(np.float32, copy=False)

    y0, x0, y1, x1 = central_roi_bounds(H, W, roi_frac=roi_frac)
    g0_roi = g0f[y0:y1, x0:x1]

    roi_bounds_internal = (0, 0, g0_roi.shape[0], g0_roi.shape[1])

    best_theta = 0.0
    best_dx, best_dy = 0, 0
    best_score = float("inf")

   # Build theta list (include endpoints)
    if theta_max <= 0:
        theta_values = [0.0]
    else:
        theta_values = np.arange(
            -theta_max,
            theta_max + 1e-9,
            theta_step,
            dtype=np.float32
        )

    for theta in theta_values:
        g1_full_rot = rotate_image_gray(g1f, float(theta))
        g1_roi_rot = g1_full_rot[y0:y1, x0:x1]

        dx, dy, score = best_translation_ssd(
            g0f=g0_roi,
            g1f=g1_roi_rot,
            max_shift_x=max_shift_x,
            max_shift_y=max_shift_y,
            roi_bounds=roi_bounds_internal
        )

        if score < best_score:
            best_score = score
            best_theta = float(theta)
            best_dx, best_dy = dx, dy

    return best_theta, best_dx, best_dy

def estimate_rigid_local(
    g0: np.ndarray, g1: np.ndarray,
    init_theta_deg: float, init_dx: int, init_dy: int,
    search_radius_xy: int, search_radius_theta_deg: float,
    theta_step_deg: float,
    roi_frac: float = 0.6,
    min_overlap: int = 20
) -> tuple[float, int, int]:
    """
    Refine a rigid transform estimate around an initial guess.

    Searches:
      theta in [init_theta_deg - search_radius_theta_deg,
               init_theta_deg + search_radius_theta_deg]
      dx    in [init_dx - search_radius_xy,
               init_dx + search_radius_xy]
      dy    in [init_dy - search_radius_xy,
               init_dy + search_radius_xy]

    Returns:
      best_theta_deg, best_dx, best_dy
    """

    H, W = g0.shape
    g0f = g0.astype(np.float32, copy=False)
    g1f = g1.astype(np.float32, copy=False)

    y0, x0, y1, x1 = central_roi_bounds(H, W, roi_frac=roi_frac)
    g0_roi = g0f[y0:y1, x0:x1]

    best_score = float("inf")
    best_theta = init_theta_deg
    best_dx = init_dx
    best_dy = init_dy

    # Build local theta range
    if search_radius_theta_deg <= 0:
        theta_values = np.array([init_theta_deg], dtype=np.float32)
    else:
        theta_values = np.arange(
            init_theta_deg - search_radius_theta_deg,
            init_theta_deg + search_radius_theta_deg + 1e-9,
            theta_step_deg,
            dtype=np.float32
        )

    # Local search
    for theta in theta_values:
        g1_full_rot = rotate_image_gray(g1f, float(theta))
        
        for dy in range(init_dy - search_radius_xy,
                        init_dy + search_radius_xy + 1):
            for dx in range(init_dx - search_radius_xy,
                            init_dx + search_radius_xy + 1):
                gy0, gx0, gy1, gx1 = y0, x0, y1, x1
                hy0 = gy0 - dy
                hx0 = gx0 - dx
                hy1 = gy1 - dy
                hx1 = gx1 - dx

                # Clip to bounds
                if hy0 < 0:
                    gy0 -= hy0
                    hy0 = 0
                if hx0 < 0:
                    gx0 -= hx0
                    hx0 = 0
                if hy1 > H:
                    overflow = hy1 - H
                    gy1 -= overflow
                    hy1 = H
                if hx1 > W:
                    overflow = hx1 - W
                    gx1 -= overflow
                    hx1 = W

                if (gy1 - gy0) < min_overlap or (gx1 - gx0) < min_overlap:
                    continue

                A = g0f[gy0:gy1, gx0:gx1]
                B = g1_full_rot[hy0:hy1, hx0:hx1]

                diff = A - B
                score = float(np.mean(diff * diff))

                if score < best_score:
                    best_score = score
                    best_theta = float(theta)
                    best_dx = dx
                    best_dy = dy

    return best_theta, best_dx, best_dy

def pyramid_estimate_rigid_from_pyramids(
    pyr0: list[np.ndarray],
    pyr1: list[np.ndarray],
    num_levels: int,
    coarse_max_shift_x: int,
    coarse_max_shift_y: int,
    coarse_theta_max_deg: float,
    coarse_theta_step_deg: float,
    refine_radius_xy: int = 2,
    refine_radius_theta_deg: float = 0.5,
    refine_theta_step_deg: float = 0.25,
    roi_frac: float = 0.4,
    min_overlap: int = 20
) -> tuple[float, int, int]:
    """
    Estimate the rigid transformation that aligns g1 to g0.

    Inputs:
        pyr0 - reference frame pyramid
        pyr1 - current frame pyramid

    Output:
        (theta, dx, dy) such that:
            g0 ≈ T(dx, dy) ∘ R(theta) ( g1 )

    That is, the returned transform maps g1 into the coordinate system of g0.
    """

    # Initial guess
    theta, dx, dy = 0.0, 0, 0

    # Iterate from coarsest (smallest image) to finest (level 0)
    for level in range(num_levels - 1, -1, -1):
        g0L = pyr0[level]
        g1L = pyr1[level]

        if level == num_levels - 1:
             # Convert level-0 max shift to the coarsest level pixel units
            scale = 2 ** level
            max_shift_x_L = max(1, int(np.ceil(coarse_max_shift_x / scale)))
            max_shift_y_L = max(1, int(np.ceil(coarse_max_shift_y / scale)))

            # --- Build ROI once on this level ---
            H, W = g0L.shape
            y0, x0, y1, x1 = central_roi_bounds(H, W, roi_frac=roi_frac)

            g0_roi = g0L[y0:y1, x0:x1]
            g1_roi = g1L[y0:y1, x0:x1]

            # --- Translation-only first (no rotation) ---
            roi_bounds_internal = (0, 0, g0_roi.shape[0], g0_roi.shape[1])

            # Score with no motion at all (baseline)
            diff0 = g0_roi - g1_roi
            score0 = float(np.mean(diff0 * diff0))  # MSE baseline

            # Best translation score
            dx_t, dy_t, scoreT = best_translation_ssd(
                g0f=g0_roi,
                g1f=g1_roi,
                max_shift_x=max_shift_x_L,
                max_shift_y=max_shift_y_L,
                roi_bounds=roi_bounds_internal
            )

            # --- Decide if we need rotation search ---
            ratio_thresh = 0.35  # threshold
            abs_thresh = 100.0   # threshold

            need_theta = (scoreT > abs_thresh) and (scoreT > ratio_thresh * max(score0, 1e-6))
            # need_theta = True
            if not need_theta:
                # Accept translation-only
                theta, dx, dy = 0.0, dx_t, dy_t
            else:
                # Do the expensive rotation+translation search only when needed
                # Global search only at coarsest level (cheap)
                theta, dx, dy = estimate_rigid(
                    g0=g0L,
                    g1=g1L,
                    max_shift_x=max_shift_x_L,
                    max_shift_y=max_shift_y_L,
                    theta_max=coarse_theta_max_deg,
                    theta_step=coarse_theta_step_deg,
                    roi_frac=roi_frac
                )
        else:
            # Scale translation guess when moving to finer level
            dx *= 2
            dy *= 2

            # If we didn't activate rotation at the coarse level, don't refine theta
            theta_radius = refine_radius_theta_deg if abs(theta) > 1e-6 else 0.0
            
            # Local refinement around current guess
            theta, dx, dy = estimate_rigid_local(
                g0=g0L,
                g1=g1L,
                init_theta_deg=theta,
                init_dx=dx,
                init_dy=dy,
                search_radius_xy=refine_radius_xy,
                search_radius_theta_deg=theta_radius,
                theta_step_deg=refine_theta_step_deg,
                roi_frac=roi_frac,
                min_overlap=min_overlap
            )

    return float(theta), int(dx), int(dy)

def central_roi_bounds(H: int, W: int, roi_frac: float = 0.6) -> tuple[int, int, int, int]:
    """
    Return bounds (y0, x0, y1, x1) for a centered ROI occupying roi_frac of the image.
    """
    if not (0.1 <= roi_frac <= 1.0):
        raise ValueError("roi_frac must be in [0.1, 1.0].")

    roi_h = max(20, int(H * roi_frac))
    roi_w = max(20, int(W * roi_frac))
    y0 = (H - roi_h) // 2
    x0 = (W - roi_w) // 2
    y1 = y0 + roi_h
    x1 = x0 + roi_w
    return y0, x0, y1, x1

def best_translation_ssd(
    g0f: np.ndarray,
    g1f: np.ndarray,
    max_shift_x: int,
    max_shift_y: int,
    roi_bounds: tuple[int, int, int, int]
) -> tuple[int, int, float]:
    """
    Find (dx, dy) that minimizes SSD between g0f and g1f using only overlap of a central ROI.

    Returns:
        best_dx, best_dy, best_score
    """
    H, W = g0f.shape
    y0, x0, y1, x1 = roi_bounds

    best_score = float("inf")
    best_dx, best_dy = 0, 0

    for dy in range(-max_shift_y, max_shift_y + 1):
        for dx in range(-max_shift_x, max_shift_x + 1):

            # ROI in g0
            gy0, gx0, gy1, gx1 = y0, x0, y1, x1

            # Corresponding ROI in g1 before shifting (x - dx, y - dy)
            hy0 = gy0 - dy
            hx0 = gx0 - dx
            hy1 = gy1 - dy
            hx1 = gx1 - dx

            # Clip to valid bounds while keeping both windows aligned
            if hy0 < 0:
                gy0 -= hy0
                hy0 = 0
            if hx0 < 0:
                gx0 -= hx0
                hx0 = 0
            if hy1 > H:
                overflow = hy1 - H
                gy1 -= overflow
                hy1 = H
            if hx1 > W:
                overflow = hx1 - W
                gx1 -= overflow
                hx1 = W

            # Skip tiny overlaps
            if (gy1 - gy0) < 20 or (gx1 - gx0) < 20:
                continue

            A = g0f[gy0:gy1, gx0:gx1]
            B = g1f[hy0:hy1, hx0:hx1]
            diff = A - B

            score = float(np.mean(diff * diff))

            if score < best_score:
                best_score = score
                best_dx, best_dy = dx, dy

    return best_dx, best_dy, best_score

#---- image transform utils ----#
def bilinear_sample_gray(g: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Bilinear sampling from grayscale image g at floating coordinates (xs, ys).

    Inputs:
        g: (H, W) float32/float64
        xs, ys: same shape arrays of float coordinates (x is column, y is row)

    Returns:
        sampled values, same shape as xs/ys, float32
    """
    H, W = g.shape

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Valid mask: coordinates must be inside [0..W-1] and [0..H-1] for bilinear neighbors
    valid = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)

    out = np.zeros(xs.shape, dtype=np.float32)
    if not np.any(valid):
        return out

    # Fractions
    ax = (xs - x0).astype(np.float32)
    ay = (ys - y0).astype(np.float32)

    # Gather 4 neighbors only where valid
    v = valid
    Ia = g[y0[v], x0[v]]
    Ib = g[y0[v], x1[v]]
    Ic = g[y1[v], x0[v]]
    Id = g[y1[v], x1[v]]

    axv = ax[v]
    ayv = ay[v]

    # Bilinear formula
    out[v] = (
        (1 - axv) * (1 - ayv) * Ia +
        axv * (1 - ayv) * Ib +
        (1 - axv) * ayv * Ic +
        axv * ayv * Id
    ).astype(np.float32)

    return out

_ROT_GRID_CACHE = {}
def rotate_image_gray(g: np.ndarray, theta_deg: float) -> np.ndarray:
    """
    Rotate a grayscale image around its center by theta_deg degrees.
    Returns float32 image (same shape).
    """
    if g.ndim != 2:
        raise ValueError("rotate_image_gray expects a 2D grayscale image.")

    g = g.astype(np.float32, copy=False)
    H, W = g.shape
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    key = (H, W)

    # 1) Take precomputed grids if we already built them for this shape
    if key in _ROT_GRID_CACHE:
        x_rel, y_rel = _ROT_GRID_CACHE[key]
    else:
        # 2) Build them once for this shape and store
        yy, xx = np.meshgrid(
            np.arange(H, dtype=np.float32),
            np.arange(W, dtype=np.float32),
            indexing="ij"
        )
        x_rel = xx - cx
        y_rel = yy - cy
        _ROT_GRID_CACHE[key] = (x_rel, y_rel)

    theta = np.deg2rad(theta_deg).astype(np.float32)
    c = np.cos(theta).astype(np.float32)
    s = np.sin(theta).astype(np.float32)

    # Inverse rotation mapping (same as your original)
    xs = c * x_rel + s * y_rel + cx
    ys = -s * x_rel + c * y_rel + cy

    out = bilinear_sample_gray(g, xs, ys)
    return out

def rigid_to_homography_center(theta_deg: float, dx: int, dy: int, W: int, H: int) -> np.ndarray:
    """
    Build a 3x3 homography for a rigid motion (rotation + translation)
    where the rotation is defined around the image center.

    The returned matrix maps points from the source frame into the
    reference frame coordinates, consistent with rotate_image_gray.

    Parameters:
        theta_deg : Rotation angle in degrees.
        dx, dy    : Translation in pixels (after rotation).
        W, H      : Width and height of the image.

    Returns:
        H : (3,3) float32 homography matrix.
    """
    # Convert degrees to radians (numpy trigonometric functions expect radians)
    theta_rad = np.deg2rad(theta_deg)

    # Rotation components
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)

    # Center of the image
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    # Build homogeneous transformation matrix
    # H = np.array([
    #     [ c, -s, dx],
    #     [ s,  c, dy],
    #     [ 0,  0,  1 ]
    # ], dtype=np.float32)

    # Adjust for rotation around center
    T1 = np.array([[1, 0, -cx],
                   [0, 1, -cy],
                   [0, 0, 1]], dtype=np.float32)
    R  = np.array([[ c, -s, 0],
                   [ s,  c, 0],
                   [ 0,  0, 1]], dtype=np.float32)
    T2 = np.array([[1, 0, cx + dx],
                   [0, 1, cy + dy],
                   [0, 0, 1]], dtype=np.float32)
    
    # return H
    return T2 @ R @ T1

#---- Pyramid and filtering utilities ----#
def gaussian_kernel_5():
    """Return a normalized 5x5 Gaussian kernel."""
    base = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    kernel_1d = base / base.sum()
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d

def convolve_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Fast 2D convolution using NumPy vectorized shifting.
    Reflect padding, supports grayscale or (H,W,C).
    """
    if image.ndim == 2:
        image = image[:, :, None]  # (H,W,1)
        added_channel = True
    else:
        added_channel = False

    image = image.astype(np.float32, copy=False)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    H, W, C = image.shape

    padded = np.pad(image, ((ph, ph), (pw, pw), (0, 0)), mode="reflect")
    out = np.zeros((H, W, C), dtype=np.float32)

    # Only 25 iterations for a 5x5 kernel (cheap)
    for r in range(kh):
        for c in range(kw):
            out += padded[r:r+H, c:c+W, :] * kernel[r, c]

    return out[:, :, 0] if added_channel else out

def reduce_image(image):
    """Reduce image size by half using Gaussian blurring and subsampling."""
    image = image.astype(np.float32)
    kernel = gaussian_kernel_5()
    blurred = convolve_2d(image, kernel)
    if blurred.ndim == 2:
        reduced = blurred[::2, ::2]
    else:
        reduced = blurred[::2, ::2, :]
    return reduced

def build_gaussian_pyramid(image, levels):
    """Build a Gaussian pyramid with the specified number of levels."""
    pyramid = [image]
    current_image = image
    for _ in range(1, levels):
        current_image = reduce_image(current_image)
        pyramid.append(current_image)
    return pyramid

#---- pipline ----#

def accumulate_rigid_transforms(pyramids: list[list[np.ndarray]],
                                frame_shape: tuple[int, int]) -> list[np.ndarray]:
    """
    pyramids[k] is the pyramid of frame k (list of levels).
    frame_shape is (H, W) of level 0 grayscale.
    """
    if pyramids is None or len(pyramids) == 0:
        raise ValueError("pyramids must be a non-empty list of pyramids.")

    H0, W0 = frame_shape

    # Choose search ranges once (based on frame size +  assumptions)
    max_shift_x, max_shift_y, theta_max, theta_step, num_levels = choose_search_ranges(frame_shape)

    # Initialize accumulation: frame 0 -> frame 0 is identity
    H_acc = np.eye(3, dtype=np.float32)
    H_list: list[np.ndarray] = [H_acc.copy()]


    for k in range(1, len(pyramids)):
        pyr_prev = pyramids[k - 1]
        pyr_curr = pyramids[k]

        # Estimate motion aligning current -> previous:
        # pyr_prev ≈ T(dx,dy) ∘ R(theta) (pyr_curr)
        theta, dx, dy = pyramid_estimate_rigid_from_pyramids(
            pyr_prev, pyr_curr,
            num_levels,
            coarse_max_shift_x=max_shift_x,
            coarse_max_shift_y=min(max_shift_y, 2),
            # coarse_max_shift_y=0,  # disable vertical shift for stability
            coarse_theta_max_deg=theta_max,
            coarse_theta_step_deg=theta_step,
            refine_radius_xy=REFINE_RADIUS_XY,
            refine_radius_theta_deg=REFINE_RADIUS_THETA,
            refine_theta_step_deg=REFINE_THETA_STEP,
            roi_frac=ROI_FRAC,
            min_overlap=MIN_OVERLAP
        )

        # theta = 0.0  # disable rotation for stability
        # dy = 0      # disable vertical shift for stability


        # (k -> 0) = (k-1 -> 0) ∘ (k -> k-1)
        H_step = rigid_to_homography_center(theta, dx, dy, W0, H0)
        H_acc = H_acc @ H_step
        H_list.append(H_acc.copy())

    return H_list


def load_frames_from_dir(input_frames_path: str) -> list[np.ndarray]:
    """
    Load all video frames from a directory.

    Args:
        input_frames_path: Path to a directory containing input frames
                           named in sequential order (e.g., frame_00000.jpg).

    Returns:
        A list of frames as NumPy arrays (H, W, 3) in BGR format.
    """
    files = sorted(
        f for f in os.listdir(input_frames_path)
        if f.endswith(".jpg") or f.endswith(".png")
    )

    frames = []
    for fname in files:
        path = os.path.join(input_frames_path, fname)
        img = cv2.imread(path)  # BGR, uint8
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        frames.append(img)

    return frames

def preprocess_all(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    Apply preprocessing to a list of frames.
    Converts each frame to grayscale and resizes it using the preprocess function.
    Args:
        frames: List of input frames as NumPy arrays (H, W, 3) in BGR format.
    Returns:
        List of preprocessed grayscale frames as float32 NumPy arrays.
    """
    if frames is None or len(frames) == 0:
        raise ValueError("frames must be a non-empty list.")

    gray_frames = []
    for frame in frames:
        gray = preprocess(frame).astype(np.float32)
        gray_frames.append(gray)

    return gray_frames

def choose_strip_positions(frame_width: int,
                           n_out_frames: int,
                           strip_width: int,
                           safety_margin_frac: float = 0.20) -> list[int]:
    """
    Choose x-start positions for vertical strips used to build panoramas.

    Args:
        frame_width: Width of the input frame in pixels.
        n_out_frames: Number of panoramas to generate (one per strip position).
        strip_width: Width of each vertical strip in pixels.
        safety_margin_frac: Fraction of width kept as margin to avoid strips leaving the frame.

    Returns:
        List of x positions (start column) of length n_out_frames.
    """
    if n_out_frames <= 0:
        raise ValueError("n_out_frames must be > 0.")
    if strip_width <= 0 or strip_width > frame_width:
        raise ValueError("strip_width must be in (0, frame_width].")

    margin = int(round(safety_margin_frac * frame_width))

    # Ensure strips fit fully inside the frame even near edges
    x_min = margin
    x_max = frame_width - margin - strip_width

    # if margin is too large, use full range
    if x_max < x_min:
        x_min = 0
        x_max = frame_width - strip_width

    if n_out_frames == 1:
        return [int((x_min + x_max) / 2)]

    xs = np.linspace(x_min, x_max, n_out_frames)
    xs = np.round(xs).astype(np.int32).tolist()
    return xs

def bilinear_sample_color(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Bilinear sampling for a color image at floating coordinates.

    Args:
        img: (H, W, 3) float32
        xs, ys: same-shape float arrays, x is column, y is row

    Returns:
        samples: (N, 3) float32 samples for valid points only
        valid: (N,) bool mask for points that are fully inside bilinear neighborhood
    """
    H, W, _ = img.shape

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)

    samples_full = np.zeros((xs.size, 3), dtype=np.float32)
    if not np.any(valid):
        return samples_full, valid

    ax = (xs - x0).astype(np.float32)
    ay = (ys - y0).astype(np.float32)

    v = valid
    Ia = img[y0[v], x0[v], :]  # (Nv,3)
    Ib = img[y0[v], x1[v], :]
    Ic = img[y1[v], x0[v], :]
    Id = img[y1[v], x1[v], :]

    axv = ax[v][:, None]  # (Nv,1)
    ayv = ay[v][:, None]

    samples = (
        (1 - axv) * (1 - ayv) * Ia +
        axv * (1 - ayv) * Ib +
        (1 - axv) * ayv * Ic +
        axv * ayv * Id
    ).astype(np.float32)

    samples_full[v, :] = samples
    return samples_full, valid

def apply_homography(H: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply homography H to points (xs, ys). Works on flat arrays.

    Args:
        H: (3,3)
        xs, ys: (N,)

    Returns:
        x2, y2: (N,) mapped coordinates (float32)
    """
    ones = np.ones_like(xs, dtype=np.float32)
    pts = np.stack([xs, ys, ones], axis=0).astype(np.float32)  # (3,N)
    q = H.astype(np.float32) @ pts
    w = q[2, :]
    # Avoid division by zero in a safe way
    w = np.where(np.abs(w) < 1e-8, 1e-8, w)
    x2 = q[0, :] / w
    y2 = q[1, :] / w
    return x2.astype(np.float32), y2.astype(np.float32)

def compute_global_canvas(frames_shape: tuple[int, int], H_list: list[np.ndarray]) -> tuple[np.ndarray, int, int]:
    """
    Compute a single canvas transform T and fixed panorama size (pano_h, pano_w)
    based on warping the FULL frame corners of all frames into frame-0 coordinates.
    Returns:
        T: (3,3) shift to move everything into positive canvas coords
        pano_h, pano_w: fixed canvas size
    """
    H0, W0 = frames_shape

    # 4 corners of the full frame
    cx = np.array([0, W0 - 1, 0, W0 - 1], dtype=np.float32)
    cy = np.array([0, 0, H0 - 1, H0 - 1], dtype=np.float32)

    all_x = []
    all_y = []

    for Hk in H_list:
        tx, ty = apply_homography(Hk, cx, cy)  # frame k -> frame 0 coords
        all_x.append(tx)
        all_y.append(ty)

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    min_x = int(np.floor(all_x.min()))
    max_x = int(np.ceil(all_x.max()))
    min_y = int(np.floor(all_y.min()))
    max_y = int(np.ceil(all_y.max()))

    pano_w = max_x - min_x + 1
    pano_h = max_y - min_y + 1

    T = np.array([[1, 0, -min_x],
                  [0, 1, -min_y],
                  [0, 0, 1]], dtype=np.float32)

    return T, pano_h, pano_w


def build_panorama_from_strips(
    frames_bgr: list[np.ndarray],
    H_shift_list: list[np.ndarray],
    H_inv_list: list[np.ndarray],
    pano_h: int,
    pano_w: int,
    x_start: int,
    strip_width: int
) -> np.ndarray:
    """
    Build one panorama on a fixed canvas by warping a vertical strip from each frame.

    Args:
        frames_bgr: List of (H0, W0, 3) uint8 BGR frames.
        H_shift_list: List of (3,3) homographies mapping frame k -> canvas coords.
        H_inv_list: List of (3,3) homographies mapping canvas -> frame k coords.
        pano_h, pano_w: Fixed output panorama height and width.
        x_start: Left x coordinate of the strip in the source frames.
        strip_width: Width of the strip in pixels.

    Returns:
        (pano_h, pano_w, 3) uint8 BGR panorama image.
    """
    if len(H_shift_list) != len(frames_bgr) or len(H_inv_list) != len(frames_bgr):
        raise ValueError("H_shift_list/H_inv_list must match frames_bgr length.")
    if len(frames_bgr) == 0:
        raise ValueError("frames_bgr must be non-empty.")
    if strip_width <= 0:
        raise ValueError("strip_width must be > 0.")

    H0, W0, C0 = frames_bgr[0].shape
    if C0 != 3:
        raise ValueError("Expected 3-channel BGR frames.")
    if x_start < 0 or x_start + strip_width > W0:
        raise ValueError("Strip is out of bounds for the input frame width.")

    # 1) Define strip corners in source frame coordinates
    x0 = float(x_start)
    x1 = float(x_start + strip_width - 1)
    y0 = 0.0
    y1 = float(H0 - 1)

    acc  = np.zeros((pano_h, pano_w, 3), dtype=np.float32)   # accumulated color
    wacc = np.zeros((pano_h, pano_w), dtype=np.float32)      # accumulated weights

    # 2) Warp each frame strip into the canvas using inverse mapping
    for k, frame in enumerate(frames_bgr):
        frame_f = frame.astype(np.float32, copy=False)

        Hk_shift = H_shift_list[k]   # frame k -> canvas
        Hk_inv   = H_inv_list[k]     # canvas -> frame k

        # Warp strip corners to canvas to get bounding box
        cx = np.array([x0, x1, x0, x1], dtype=np.float32)
        cy = np.array([y0, y0, y1, y1], dtype=np.float32)
        tx, ty = apply_homography(Hk_shift, cx, cy)

        bx0 = int(max(0, np.floor(tx.min())))
        bx1 = int(min(pano_w - 1, np.ceil(tx.max())))
        by0 = int(max(0, np.floor(ty.min())))
        by1 = int(min(pano_h - 1, np.ceil(ty.max())))

        if bx1 < bx0 or by1 < by0:
            continue

        # Canvas grid
        xs = np.arange(bx0, bx1 + 1, dtype=np.float32)
        ys = np.arange(by0, by1 + 1, dtype=np.float32)
        Xc, Yc = np.meshgrid(xs, ys, indexing="xy")

        Xc_flat = Xc.ravel()
        Yc_flat = Yc.ravel()

        # Map canvas -> source frame
        xs_src, ys_src = apply_homography(Hk_inv, Xc_flat, Yc_flat)

        # Keep only points inside the strip
        in_strip = (
            (xs_src >= x0) & (xs_src <= x1) &
            (ys_src >= y0) & (ys_src <= y1)
        )

        samples_full, valid_bilin = bilinear_sample_color(frame_f, xs_src, ys_src)
        valid = in_strip & valid_bilin

        if not np.any(valid):
            continue

        idx = np.where(valid)[0]
        Xd = Xc_flat[idx].astype(np.int32)
        Yd = Yc_flat[idx].astype(np.int32)

        feather = max(1.0, 0.35 * strip_width)  
        d = np.minimum(xs_src[idx] - x0, x1 - xs_src[idx])  
        w = np.clip(d / feather, 0.05, 1.0).astype(np.float32)
        w = w * w * (3 - 2 * w)   # smoothstep


        acc[Yd, Xd, :] += samples_full[idx, :] * w[:, None]
        wacc[Yd, Xd]   += w

    out = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
    mask = wacc > 1e-6
    out[mask, :] = acc[mask, :] / wacc[mask, None]
    pano_u8 = np.clip(out, 0, 255).astype(np.uint8)

    return pano_u8

def generate_panorama(input_frames_path, n_out_frames):
    """
    Main entry point 
    :param input_frames_path : path to a dir with input video frames.
    :param n_out_frames: number of generated panorama frames
    :return: A list of generated panorma frames (of size n_out_frames),
    each list item should be a PIL image of a generated panorama.
    """
    # 1) Read all frames from directory (sorted)
    frames_bgr = load_frames_from_dir(input_frames_path)

    if len(frames_bgr) == 0:
        raise ValueError("No frames found in input directory.")

    # 2) Preprocess frames (gray, resize)
    gray_frames = preprocess_all(frames_bgr)

    # choose pyramid params once
    max_shift_x, max_shift_y, theta_max, theta_step, num_levels = choose_search_ranges(gray_frames[0].shape)

    # build pyramids once per frame
    pyramids = [
        build_gaussian_pyramid(g.astype(np.float32, copy=False), num_levels)
        for g in gray_frames
    ]

    # 3) Compute accumulated transforms H_list
    frame_shape = gray_frames[0].shape
    H_list = accumulate_rigid_transforms(pyramids, frame_shape)

    T, pano_h, pano_w = compute_global_canvas(frames_bgr[0].shape[:2], H_list)

    H_shift_list = [T @ Hk for Hk in H_list]
    H_inv_list = [np.linalg.inv(Hs).astype(np.float32) for Hs in H_shift_list]

    # 4) Choose x-positions for strips (n_out_frames)
    frame_height, frame_width = frames_bgr[0].shape[:2]
    strip_x_positions = choose_strip_positions(frame_width, n_out_frames, STRIP_WIDTH)

    # 5) Build panoramas
    panoramas = []
    for x_start in strip_x_positions:
        # Build one panorama from strips at x_start
        pano_bgr = build_panorama_from_strips(frames_bgr,
                                                H_shift_list,
                                                H_inv_list,
                                                pano_h,
                                                pano_w,
                                                x_start,
                                                STRIP_WIDTH)
        
        # Convert BGR -> RGB -> PIL
        pano_rgb = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB)
        pano_pil = Image.fromarray(pano_rgb)
        panoramas.append(pano_pil)


    # 6) Convert panoramas to PIL and return
    return panoramas



def main():
   pass

if __name__ == "__main__":
    main()