# Sparse Pointcloud Generator

## Overview

This project implements a Structure from Motion (SfM) based sparse pointcloud generator using OpenCV, Open3D, PyTorch, and COLMAP. The tool can extract frames from a video, detect features, match them across consecutive frames, estimate camera motion, and generate a 3D point cloud representation.

## Features

- Video frame extraction with flexible sampling
- Multiple feature detection methods (SIFT, ORB)
- Feature matching and tracking
- Camera motion estimation
- 3D point cloud generation using Open3D or COLMAP
- Verbose logging and progress tracking
- Configurable frame sampling rate

## Prerequisites

- Conda (Anaconda or Miniconda)
- GPU support recommended (optional)

## Installation

### Open3D-based SfM Pipeline

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sparse-pointcloud-generator.git
cd sparse-pointcloud-generator
```

2. Create a Conda environment:
```bash
conda create -n sfm_env python=3.9
conda activate sfm_env
```

3. Install dependencies:
```bash
conda install -c conda-forge opencv
conda install -c conda-forge open3d
conda install pytorch torchvision torchaudio -c pytorch
conda install -c conda-forge kornia
pip install tqdm
```

### COLMAP-based SfM Pipeline

1. Install COLMAP:
   - On Ubuntu:
   ```bash
   sudo apt-get install colmap
   ```
   - On macOS (via Homebrew):
   ```bash
   brew install colmap
   ```
   - On Windows, download from [COLMAP's official site](https://colmap.github.io/)

2. Install `pycolmap`:
```bash
pip install pycolmap
```

## Usage

### Open3D-based SfM
```python
from sparse_pointcloud_generator import SparsePointCloudGenerator

# Initialize generator
video_path = 'path/to/your/video.mp4'
sfm_generator = SparsePointCloudGenerator(video_path, verbose=True)

# Generate point cloud (sampling at 2 fps)
point_cloud = sfm_generator.generate_pointcloud(target_fps=2)

# Visualize point cloud
import open3d as o3d
o3d.visualization.draw_geometries([point_cloud])
```

### COLMAP-based SfM
```python
import pathlib
from pycolmap_sfm.sfm_colmap import run_sfm

image_dir = pathlib.Path("path/to/images")
output_path = pathlib.Path("path/to/output")

run_sfm(image_dir, output_path, use_gpu=False)
```

## Configuration Options

- `verbose`: Enable/disable detailed logging
- `target_fps`: Control frame sampling rate for Open3D pipeline
- `method`: Feature detection method ('sift' or 'orb')
- `use_gpu`: Enable/disable GPU processing for COLMAP pipeline

## Performance Tips

- Use GPU-enabled OpenCV and COLMAP for faster processing
- Adjust `target_fps` to balance accuracy and computational cost
- Experiment with different feature detection methods

## Troubleshooting

- Ensure all dependencies are correctly installed
- Check video file compatibility
- Verify GPU drivers if using GPU acceleration

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- OpenCV
- Open3D
- PyTorch
- Kornia
- COLMAP
- Pycolmap

## Citation

If you use this project in your research, please cite:
```
@misc{sparse_pointcloud_generator,
  title={Sparse Pointcloud Generator},
  author={Kumar Ankit},
  year={2025},
  url={https://github.com/krankit00/sfm_project}
}
```
