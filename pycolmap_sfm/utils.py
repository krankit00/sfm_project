import pycolmap
import os
import cv2
import open3d as o3d

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

def process_pointcloud(output_path, view_pcd: bool = True, save_pcd: bool = True, filter_pcd: bool = True):
    # print(f"Loading sparse reconstruction from {output_path / "0"}...")
    reconstruction = pycolmap.Reconstruction(output_path / "0" )
    if not os.path.isdir(output_path / "pointcloud.ply") and save_pcd:
        print("Pointcloud Saved")
        reconstruction.export_PLY(output_path / "pointcloud.ply")    
    pcd = o3d.io.read_point_cloud(output_path / "pointcloud.ply")
    if filter_pcd:
        pcd = statistical_outlier_removal(pcd)
        if save_pcd:
            o3d.io.write_point_cloud(output_path / "pointcloud_filtered.ply", pcd)
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