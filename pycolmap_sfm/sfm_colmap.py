import pathlib
import pycolmap

def run_sfm(image_dir: pathlib.Path, output_path: pathlib.Path, use_gpu: bool = False):
    """Runs Structure-from-Motion (SfM) using pycolmap with a configurable GPU flag."""
    output_path.mkdir(parents=True, exist_ok=True)
    mvs_path = output_path / "mvs"
    database_path = output_path / "database.db"
    
    print("Extracting features...")
    pycolmap.extract_features(database_path, image_dir, use_gpu=use_gpu)
    
    print("Matching features...")
    pycolmap.match_exhaustive(database_path, use_gpu=use_gpu)
    
    print("Running incremental mapping...")
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path, use_gpu=use_gpu)
    if not maps:
        raise RuntimeError("Incremental mapping failed: No valid maps generated.")
    
    print("Saving reconstructed map...")
    maps[0].write(output_path)
    
    # Dense reconstruction
    print("Undistorting images...")
    pycolmap.undistort_images(mvs_path, output_path, image_dir)
    
    print("Running PatchMatch Stereo...")
    pycolmap.patch_match_stereo(mvs_path, use_gpu=use_gpu)  # Requires compilation with CUDA if GPU is used
    
    print("Fusing stereo depth maps...")
    pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)
    
    print("SfM pipeline completed successfully.")


# Example usage:
run_sfm(pathlib.Path("images"), pathlib.Path("output"))
