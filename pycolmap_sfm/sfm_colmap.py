import pathlib
import pycolmap
import os
import cv2
import numpy as np
import open3d as o3d
import tqdm

def visualize_sparse_pointcloud(output_path):
    sparse_dir = output_path / "0"  # COLMAP's default output directory
    points3D_file = sparse_dir / "points3D.bin"

    if not points3D_file.exists():
        raise FileNotFoundError(f"Sparse point cloud file not found: {points3D_file}")

    print(f"Loading sparse reconstruction from {points3D_file}...")
    reconstruction = pycolmap.Reconstruction(sparse_dir)
    if not os.path.isdir(output_path / "pointcloud.ply"):
        print("Pointcloud Saved")
        reconstruction.export_PLY(output_path / "pointcloud.ply")

    pcd = o3d.io.read_point_cloud(output_path / "pointcloud.ply")
    print("Visualizing sparse point cloud with color texture...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Sparse Point Cloud", width=1280, height=720)

    # Add the point cloud
    vis.add_geometry(pcd)

    # Customize view controls
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)  # Zoom level
    ctr.set_front([0, 0, -1])  # Front view
    ctr.set_lookat([0, 0, 0])  # Look at the origin
    ctr.set_up([0, -1, 0])  # Camera up direction

    # Run the visualizer
    vis.run()
    vis.destroy_window()

def extract_frames(video_path, output_dir, fps):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    frame_count = 0
    success, frame = cap.read()
    while success:
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
        frame_count += 1
        success, frame = cap.read()
    cap.release()

def run_sfm(image_dir: pathlib.Path, output_path: pathlib.Path, use_gpu: bool = False):
    """Runs Structure-from-Motion (SfM) using pycolmap with a configurable GPU flag."""
    output_path.mkdir(parents=True, exist_ok=True)
    mvs_path = output_path / "mvs"
    database_path = output_path / "database.db"
    
    print("Extracting features...")
    pycolmap.extract_features(database_path, image_dir)
    
    print("Matching features...")
    # pycolmap.match_exhaustive(database_path)
    pycolmap.match_sequential(database_path)
        
    print("Running incremental mapping...")
    if not os.path.isdir(output_path / "0"):
        maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
        if not maps:
            raise RuntimeError("Incremental mapping failed: No valid maps generated.")
        else:
            print("Saving reconstructed map...")
            maps[0].write(output_path)
    else:
       print("Already reconstructed. Loading reconstruction!")
    
    print("SfM pipeline completed successfully.")

# Example usage:
video_path = "../watchtower.mp4"
image_dir = pathlib.Path("extracted_frames")
output_path = pathlib.Path("output")
extract_frames(video_path, image_dir, fps=1)

maps = run_sfm(image_dir=image_dir,output_path=output_path)
visualize_sparse_pointcloud(output_path)
