import pathlib
import pycolmap
import os
import cv2
import numpy as np
import open3d as o3d
import tqdm
import signal
import sys

def statistical_outlier_removal(pcd, nb_neighbors=50, std_ratio=1.0):
    """
    Statistical outlier removal filter
    Removes points that are statistical outliers from the neighborhood
    """
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    
    filtered_pcd = pcd.select_by_index(ind)
    return filtered_pcd

def visualize_sparse_pointcloud(output_path, apply_filter=False, nb_neighbors=50, std_ratio=1.0):
    sparse_dir = output_path / "0"  # COLMAP's default output directory
    points3D_file = sparse_dir / "points3D.bin"

    if not points3D_file.exists():
        raise FileNotFoundError(f"Sparse point cloud file not found: {points3D_file}")

    print(f"Loading sparse reconstruction from {points3D_file}...")
    reconstruction = pycolmap.Reconstruction(sparse_dir)
    
    points = np.array([reconstruction.points3D[p_id].xyz for p_id in reconstruction.points3D])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if apply_filter:
        pcd = statistical_outlier_removal(pcd, nb_neighbors, std_ratio)

    o3d.visualization.draw_geometries([pcd], window_name="Sparse Point Cloud")

def visualize_camera_poses(output_path):
    sparse_dir = output_path / "0"
    images_file = sparse_dir / "images.bin"

    if not images_file.exists():
        raise FileNotFoundError(f"Camera poses file not found: {images_file}")

    print(f"Loading camera poses from {images_file}...")
    reconstruction = pycolmap.Reconstruction(sparse_dir)

    camera_positions = []
    for image in reconstruction.images.values():
        camera_positions.append(image.projection_center())
    
    camera_positions = np.array(camera_positions)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(camera_positions)
    pcd.paint_uniform_color([1, 0, 0])  # Red color for cameras

    o3d.visualization.draw_geometries([pcd], window_name="Camera Poses")

def run_sfm(image_path, output_path, matcher="sequential"):
    options = {
        "exhaustive": pycolmap.ExhaustiveMatchingOptions(),
        "sequential": pycolmap.SequentialMatchingOptions(),
        "vocab_tree": pycolmap.VocabTreeMatchingOptions()
    }
    
    if matcher not in options:
        raise ValueError(f"Invalid matcher option. Available: {list(options.keys())}")
    
    print(f"Running SfM with {matcher} matcher...")
    with pycolmap.automatic_reconstructor(image_path, output_path, matching_options=options[matcher]) as reconstructor:
        reconstructor.run()

def signal_handler(sig, frame):
    print("\nProcess interrupted. Cleaning up...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Structure-from-Motion with Pycolmap")
    parser.add_argument("image_path", type=pathlib.Path, help="Path to images")
    parser.add_argument("output_path", type=pathlib.Path, help="Output directory")
    parser.add_argument("--matcher", type=str, choices=["exhaustive", "sequential", "vocab_tree"], default="exhaustive", help="Feature matching method")
    parser.add_argument("--filter", action="store_true", help="Apply statistical filtering to the point cloud")
    parser.add_argument("--nb_neighbors", type=int, default=50, help="Number of neighbors for filtering")
    parser.add_argument("--std_ratio", type=float, default=1.0, help="Standard deviation ratio for filtering")
    parser.add_argument("--vis_cameras", action="store_true", help="Visualize camera poses")

    args = parser.parse_args()

    run_sfm(args.image_path, args.output_path, matcher=args.matcher)
    visualize_sparse_pointcloud(args.output_path, apply_filter=args.filter, nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio)
    
    if args.vis_cameras:
        visualize_camera_poses(args.output_path)
