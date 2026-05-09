# stereo-mosaicing-panorama

# Stereo Mosaicing Panorama Generation

Computer vision pipeline for generating panoramas from video sequences using hierarchical direct image alignment and strip mosaicing.

## Features
- SSD-based direct image alignment
- Gaussian pyramid coarse-to-fine motion estimation
- Rigid transformation accumulation
- Inverse warping + bilinear interpolation
- Strip-based panorama generation
- Feathering-based blending

## Example Results
![Result](result.gif)

## Running

```bash
pip install -r requirements.txt

```python
from stereo_mosaicing import generate_panorama

panoramas = generate_panorama(
    input_frames_path="input_frames/",
    n_out_frames=5
)
```
