import pathlib
import pycolmap
import os


def run_sfm(image_dir: pathlib.Path, output_path: pathlib.Path, use_gpu: bool = False,  save_pcd: bool = True):
    """Runs Structure-from-Motion (SfM) using pycolmap with a configurable GPU flag."""
    output_path.mkdir(parents=True, exist_ok=True)
    mvs_path = output_path / "mvs"
    database_path = output_path / "database.db"
    sparse_dir = output_path / "0"  # COLMAP's default output directory
    points3D_file = sparse_dir / "points3D.bin"
    
    print("Running incremental mapping...")
    if not os.path.isdir(output_path / "0"):
        print("Extracting features...")
        pycolmap.extract_features(database_path, image_dir)
        print("Matching features...")
        # pycolmap.match_exhaustive(database_path)
        pycolmap.match_sequential(database_path)
        maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
        if not maps:
            raise RuntimeError("Incremental mapping failed: No valid maps generated.")
        else:
            print("Saving reconstructed map...")
            maps[0].write(output_path)
    else:
       print("Already reconstructed. Loading reconstruction!")
    
    
    
    print("SfM pipeline completed successfully.")
