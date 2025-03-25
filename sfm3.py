import cv2
import numpy as np
import open3d as o3d
import logging
from tqdm import tqdm

class FeatureDetector:
    """
    Class to manage different feature detection methods
    """
    @staticmethod
    def orb_detector(frame):
        """
        ORB (Oriented FAST and Rotated BRIEF) feature detector
        Fast and computationally efficient
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.ORB_create()
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        return keypoints, descriptors, 'ORB'
    
    @staticmethod
    def sift_detector(frame):
        """
        SIFT (Scale-Invariant Feature Transform) detector
        More robust to scale and rotation changes
        """
        # Ensure OpenCV contrib is installed for SIFT
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.SIFT_create()
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        return keypoints, descriptors, 'SIFT'
    
    @staticmethod
    def surf_detector(frame):
        """
        SURF (Speeded Up Robust Features) detector
        Fast alternative to SIFT
        """
        # Note: SURF is patented and not available in recent OpenCV versions
        # This implementation uses a custom approach
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.xfeatures2d.SURF_create()
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        return keypoints, descriptors, 'SURF'

class PointCloudFilter:
    """
    Class to manage different point cloud filtering techniques
    """
    @staticmethod
    def statistical_outlier_removal(point_cloud, nb_neighbors=20, std_ratio=1.0):
        """
        Statistical outlier removal filter
        Removes points that are statistical outliers from the neighborhood
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        
        filtered_pcd = pcd.select_by_index(ind)
        return filtered_pcd
    
    @staticmethod
    def radius_outlier_removal(point_cloud, nb_points=16, radius=0.05):
        """
        Radius outlier removal filter
        Removes points that have fewer neighbors within a given radius
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        
        cl, ind = pcd.remove_radius_outlier(
            nb_points=nb_points,
            radius=radius
        )
        
        filtered_pcd = pcd.select_by_index(ind)
        return filtered_pcd

def extract_frames_by_fps(video_path, target_fps=1):
    """
    Extract frames from video based on target FPS
    
    Args:
    - video_path (str): Path to the input video
    - target_fps (float): Number of frames to extract per second
    
    Returns:
    - List of frames
    - Original video FPS
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame skip
    frame_skip = max(1, int(original_fps / target_fps))
    
    # Extract frames
    frames = []
    frame_count = 0
    
    logging.info(f"Video FPS: {original_fps}")
    logging.info(f"Target FPS: {target_fps}")
    logging.info(f"Frame skip: {frame_skip}")
    
    with tqdm(total=total_frames, desc="Extracting Frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified intervals
            if frame_count % frame_skip == 0:
                frames.append(frame)
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    
    logging.info(f"Total frames extracted: {len(frames)}")
    return frames, original_fps

def structure_from_motion(frames, feature_detector=None, camera_matrix=None):
    """
    Perform Structure from Motion on extracted frames
    
    Args:
    - frames (list): List of video frames
    - feature_detector (function): Feature detection method
    - camera_matrix (np.array): Camera intrinsic matrix
    
    Returns:
    - Camera poses
    - Point cloud
    """
    # Default feature detector
    if feature_detector is None:
        feature_detector = FeatureDetector.orb_detector
    
    # Default camera matrix
    if camera_matrix is None:
        camera_matrix = np.array([
            [700, 0, 320],  # fx, skew, cx
            [0, 700, 240],  # 0, fy, cy
            [0, 0, 1]       # 0, 0, 1
        ])
    
    # Lists to store results
    camera_poses = [(np.eye(3), np.zeros((3, 1)))]
    point_cloud = []
    
    # Extract features from first frame
    prev_kp, prev_desc, detector_name = feature_detector(frames[0])
    logging.info(f"Using {detector_name} feature detector")
    
    with tqdm(total=len(frames)-1, desc="Processing Frames") as pbar:
        for curr_frame in frames[1:]:
            # Extract features from current frame
            curr_kp, curr_desc, _ = feature_detector(curr_frame)
            
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(prev_desc, curr_desc)
            matches = sorted(matches, key=lambda x: x.distance)[:50]
            
            # Get matched keypoints
            matched_prev_kp = [prev_kp[m.queryIdx] for m in matches]
            matched_curr_kp = [curr_kp[m.trainIdx] for m in matches]
            
            # Convert keypoints to points
            points1 = np.float32([kp.pt for kp in matched_prev_kp])
            points2 = np.float32([kp.pt for kp in matched_curr_kp])
            
            # Estimate essential matrix
            E, _ = cv2.findEssentialMat(points1, points2, camera_matrix)
            
            # Recover pose
            _, R, t, _ = cv2.recoverPose(E, points1, points2, camera_matrix)
            
            # Update camera poses
            prev_R, prev_t = camera_poses[-1]
            new_R = prev_R @ R
            new_t = prev_t + prev_R @ t
            camera_poses.append((new_R, new_t))
            
            # Triangulate points
            for kp1, kp2 in zip(matched_prev_kp, matched_curr_kp):
                point_3d = cv2.triangulatePoints(
                    camera_matrix @ np.hstack((prev_R, prev_t)), 
                    camera_matrix @ np.hstack((new_R, new_t)), 
                    kp1.pt, 
                    kp2.pt
                )
                point_3d /= point_3d[3]  # Homogeneous to cartesian
                point_cloud.append(point_3d[:3])
            
            # Update previous frame info
            prev_kp = curr_kp
            prev_desc = curr_desc
            
            pbar.update(1)
    
    return camera_poses, point_cloud

def visualize_point_cloud(point_cloud, camera_poses, filtered_point_cloud=None):
    """
    Visualize point cloud and camera poses using Open3D
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SFM Reconstruction")
    
    # Original point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.paint_uniform_color([0, 0, 1])  # Blue
    vis.add_geometry(pcd)
    
    # Filtered point cloud
    if filtered_point_cloud is not None:
        vis.add_geometry(filtered_point_cloud)
    
    # Add coordinate frames for camera poses
    for R, t in camera_poses:
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=t.flatten()
        )
        vis.add_geometry(camera_frame)
    
    # Coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(coord_frame)
    
    # Run visualizer
    vis.run()
    vis.destroy_window()

def main(video_path, target_fps=1):
    """
    Main SFM pipeline
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Extract frames
    frames, original_fps = extract_frames_by_fps(video_path, target_fps)
    
    # Available feature detectors
    feature_detectors = [
        FeatureDetector.orb_detector,
        # FeatureDetector.sift_detector,
        # Uncomment if OpenCV contrib is installed
        # FeatureDetector.surf_detector
    ]
    
    # Available point cloud filters
    point_cloud_filters = [
        # PointCloudFilter.statistical_outlier_removal,
        PointCloudFilter.radius_outlier_removal
    ]
    
    # Run SFM for each feature detector
    for detector in feature_detectors:
        logging.info(f"Running SFM with {detector.__name__}")
        
        # Perform SFM
        camera_poses, point_cloud = structure_from_motion(
            frames, 
            feature_detector=detector
        )
        
        # Run filters
        for filter_method in point_cloud_filters:
            logging.info(f"Applying {filter_method.__name__}")
            
            # Filter point cloud
            filtered_pcd = filter_method(point_cloud)
            
            # Visualize results
            visualize_point_cloud(point_cloud, camera_poses, filtered_pcd)
            
            # Save filtered point cloud
            output_path = f'filtered_pointcloud_{detector.__name__}_{filter_method.__name__}.ply'
            o3d.io.write_point_cloud(output_path, filtered_pcd)
            logging.info(f"Saved point cloud to {output_path}")

# Example usage
if __name__ == "__main__":
    video_path = 'watchtower.mp4'
    main(video_path, target_fps=5)