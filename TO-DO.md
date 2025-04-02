# Project TO-DO List

## High Priority
- [ ] Finalize Structure-from-Motion (SfM) pipeline using `pycolmap` and `open3d`
- [ ] Ensure proper visualization of sparse point clouds with camera poses
- [ ] Add controls for Open3D visualization (e.g., wireframe toggle, point size adjustment)
- [ ] Implement configurable GPU usage toggle for `pycolmap`
- [ ] Optimize feature extraction and matching in `pycolmap`
- [ ] Ensure proper handling of COLMAP standard output files (`cameras.bin`, `images.bin`, `points3D.bin`)
- [ ] MVS Integration

## Medium Priority
- [ ] Enhance video frame extraction logic (handle various FPS rates, video formats)
- [ ] Improve error handling for missing or corrupt COLMAP reconstruction files
- [ ] Add automated testing for the SfM pipeline
- [ ] Document `sfm_colmap.py` script with clear usage instructions
- [ ] Implement parallel processing for feature extraction and matching

## Low Priority
- [ ] Explore integration of additional SfM methods (e.g., global SfM)
- [ ] Implement a GUI-based visualization tool
- [ ] Add support for multi-camera reconstructions
- [ ] Enhance point cloud visualization with depth-based color mapping
- [ ] Optimize performance for large-scale reconstructions

## Documentation & Cleanup
- [ ] Update `README.md` to include both Open3D and COLMAP-based pipelines
- [ ] Provide installation instructions for dependencies (Open3D, COLMAP, Pycolmap, OpenCV, etc.)
- [ ] Refactor code for better modularity and maintainability
- [ ] Add inline comments and docstrings for all functions
- [ ] Ensure cross-platform compatibility (Windows, Linux, macOS)
