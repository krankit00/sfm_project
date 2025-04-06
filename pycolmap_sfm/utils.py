import torch
import pycolmap
import os
import cv2
import open3d as o3d
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed, parallel_backend

def torch_sor_chunked(pcd, nb_neighbors=5000, std_ratio=0.1, device='cuda', batch_size=1000): # Tune batch_size for GPU Memory failures
    pts = torch.tensor(np.asarray(pcd.points), device=device)
    N = pts.shape[0]
    mean_dists = torch.empty(N, device=device)

    for i in tqdm(range(0, N, batch_size), desc="Chunked SOR"):
        end = min(i + batch_size, N)
        batch = pts[i:end]  # (B, 3)
        dists = torch.cdist(batch, pts, p=2)  # (B, N)
        topk, _ = torch.topk(dists, nb_neighbors + 1, largest=False)  # (B, k+1)
        mean_dists[i:end] = topk[:, 1:].mean(dim=1)  # Exclude self

    mean = mean_dists.mean()
    std = mean_dists.std()
    threshold = mean + std_ratio * std
    inliers = (mean_dists < threshold).nonzero(as_tuple=True)[0].cpu().numpy()
    return pcd.select_by_index(inliers.tolist())

def custom_statistical_outlier_removal_parallel(pcd, nb_neighbors=5000, std_ratio=0.1, n_jobs=10):
    pts = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    def compute_mean_dist(i):
        [_, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], nb_neighbors)
        neighbors = pts[idx, :]
        dists = np.linalg.norm(neighbors - pts[i], axis=1)
        return np.mean(dists)

    with parallel_backend("threading"):
        mean_dists = Parallel(n_jobs=n_jobs)(
            delayed(compute_mean_dist)(i) for i in tqdm(range(len(pts)), desc="Threaded distance computation")
        )

    mean = np.mean(mean_dists)
    std = np.std(mean_dists)
    threshold = mean + std_ratio * std

    inliers = [i for i, dist in enumerate(mean_dists) if dist < threshold]
    return pcd.select_by_index(inliers)


def statistical_outlier_removal(pcd, nb_neighbors=5000, std_ratio=0.1):
        """
        Statistical outlier removal filter
        Removes points that are statistical outliers from the neighborhood
        """
        # pcd = o3d.io.read_point_cloud(output_path / "pointcloud.ply")
        
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        
        filtered_pcd = pcd.select_by_index(ind)
        print(f"Filtered Points: {len(pcd.points) - len(ind)} out of {len(pcd.points)}")
        return filtered_pcd

def visualize_sparse_pointcloud(pcd):
    
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

def process_pointcloud(output_path, view_pcd: bool = True, save_pcd: bool = True, filter_pcd: bool = True, is_dense: bool = False):
    # print(f"Loading sparse reconstruction from {output_path / "0"}...")
    if not is_dense:
        reconstruction = pycolmap.Reconstruction(output_path / "0" )
        if not os.path.exists(output_path / "pointcloud.ply") and save_pcd:
            print("Pointcloud Saved")
            reconstruction.export_PLY(output_path / "pointcloud.ply")    
        pcd = o3d.io.read_point_cloud(output_path / "pointcloud.ply")
    else:
        pcd = o3d.io.read_point_cloud(output_path / "mvs" / "dense.ply")
    if filter_pcd:
        if not os.path.exists(output_path / "pointcloud_filtered.ply"):
            print("Performing Outlier Removal ...")
            pcd = torch_sor_chunked(pcd)
            if save_pcd:
                o3d.io.write_point_cloud(output_path / "pointcloud_filtered.ply", pcd)
        else:
            print("Filtered Pointcloud Exists. Loading from ...", output_path / "pointcloud_filtered.ply")
            pcd = o3d.io.read_point_cloud(output_path / "pointcloud_filtered.ply")
    if view_pcd:
        visualize_sparse_pointcloud(pcd)

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
    print("Total Frame Extracted: ", frame_count)
    cap.release()