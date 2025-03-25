import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_features(frame):
    """
    Extract features from a frame using ORB detector
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create ORB detector
    orb = cv2.ORB_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def match_features(desc1, desc2):
    """
    Match features between two frames
    """
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(desc1, desc2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    return matches

def estimate_motion(matched_kp1, matched_kp2, camera_matrix):
    """
    Estimate camera motion between two frames
    """
    # Convert keypoints to numpy array
    points1 = np.float32([kp.pt for kp in matched_kp1])
    points2 = np.float32([kp.pt for kp in matched_kp2])
    
    # Estimate essential matrix
    E, mask = cv2.findEssentialMat(points1, points2, camera_matrix)
    
    # Recover pose (rotation and translation)
    _, R, t, _ = cv2.recoverPose(E, points1, points2, camera_matrix)
    
    return R, t

def structure_from_motion(video_path, max_frames=None):
    """
    Perform Structure from Motion on a video
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    # Camera matrix (assuming default intrinsics)
    camera_matrix = np.array([
        [700, 0, 320],  # fx, skew, cx
        [0, 700, 240],  # 0, fy, cy
        [0, 0, 1]       # 0, 0, 1
    ])
    
    # List to store camera poses
    camera_poses = []
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        logger.error("Could not read video")
        return
    
    # Extract features from first frame
    prev_kp, prev_desc = extract_features(prev_frame)
    
    # Initial pose
    camera_poses.append((np.eye(3), np.zeros((3, 1))))
    
    # 3D point cloud
    point_cloud = []
    
    # Progress bar
    pbar = tqdm(total=total_frames, desc="Processing Frames")
    
    frame_count = 0
    while frame_count < total_frames:
        # Read next frame
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        # Extract features from current frame
        curr_kp, curr_desc = extract_features(curr_frame)
        
        # Match features
        matches = match_features(prev_desc, curr_desc)
        
        # Select good matches
        good_matches = matches[:100]  # Adjust as needed
        
        # Get matched keypoints
        matched_prev_kp = [prev_kp[m.queryIdx] for m in good_matches]
        matched_curr_kp = [curr_kp[m.trainIdx] for m in good_matches]
        
        # Estimate motion
        R, t = estimate_motion(matched_prev_kp, matched_curr_kp, camera_matrix)
        
        # Update camera poses
        prev_R, prev_t = camera_poses[-1]
        new_R = prev_R @ R
        new_t = prev_t + prev_R @ t
        camera_poses.append((new_R, new_t))
        
        # Triangulate points (simplified)
        for kp1, kp2 in zip(matched_prev_kp, matched_curr_kp):
            # Triangulation logic (simplified)
            point_3d = cv2.triangulatePoints(
                camera_matrix @ np.hstack((prev_R, prev_t)), 
                camera_matrix @ np.hstack((new_R, new_t)), 
                kp1.pt, 
                kp2.pt
            )
            point_3d /= point_3d[3]  # Homogeneous to cartesian
            point_cloud.append(point_3d[:3])
        
        # Update previous frame information
        prev_frame = curr_frame
        prev_kp = curr_kp
        prev_desc = curr_desc
        
        # Update progress
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Log point cloud information
    logger.info(f"Total points generated: {len(point_cloud)}")
    
    return camera_poses, point_cloud

def filter_point_cloud(point_cloud):
    """
    Filter point cloud using statistical outlier removal
    """
    # Convert point cloud to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    logger.info("Performing Statistical Outlier Removal")
    # Statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=30,  # Number of neighbors to analyze for each point
        std_ratio=1.0     # Standard deviation threshold
    )
    
    # Filter point cloud
    filtered_pcd = pcd.select_by_index(ind)
    
    logger.info(f"Points before filtering: {len(pcd.points)}")
    logger.info(f"Points after filtering: {len(filtered_pcd.points)}")
    
    return filtered_pcd

def visualize_sfm_results(camera_poses, point_cloud, filtered_point_cloud=None):
    """
    Visualize camera poses and 3D point cloud using Open3D
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Structure from Motion Reconstruction")
    
    # Add original point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.paint_uniform_color([0, 0, 1])  # Blue for original point cloud
    vis.add_geometry(pcd)
    
    # Add filtered point cloud if available
    if filtered_point_cloud is not None:
        filtered_pcd = filtered_point_cloud
        filtered_pcd.paint_uniform_color([1, 0, 0])  # Red for filtered point cloud
        vis.add_geometry(filtered_pcd)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
    )
    vis.add_geometry(coord_frame)
    
    # Add camera poses as coordinate frames
    for R, t in camera_poses:
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=t.flatten()
        )
        vis.add_geometry(camera_frame)
    
    # Set up view control
    view_control = vis.get_view_control()
    view_control.set_zoom(0.5)
    
    # Update and run visualizer
    vis.run()
    vis.destroy_window()

def main(video_path, max_frames=None):
    """
    Main function to run SFM pipeline
    """
    # Perform Structure from Motion
    camera_poses, point_cloud = structure_from_motion(video_path, max_frames)
    
    # Filter point cloud
    filtered_point_cloud = filter_point_cloud(point_cloud)
    
    # Save filtered point cloud as PLY
    output_ply_path = 'filtered_point_cloud.ply'
    o3d.io.write_point_cloud(output_ply_path, filtered_point_cloud)
    logger.info(f"Filtered point cloud saved to {output_ply_path}")
    o3d.visualization.draw_geometries([filtered_point_cloud])
    
    # Visualize results
    # visualize_sfm_results(camera_poses, point_cloud, filtered_point_cloud)
    

# Example usage
if __name__ == "__main__":
    video_path = 'watchtower.mp4'
    main(video_path, max_frames=0)  # Limit to 100 frames for demonstration