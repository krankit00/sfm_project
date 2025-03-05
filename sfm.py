import numpy as np
import cv2
import open3d as o3d
import kornia
import torch
import logging
from tqdm import tqdm

class SparsePointCloudGenerator:
    def __init__(self, video_path, verbose=True, log_level=logging.INFO):
        """
        Initialize the sparse pointcloud generator
        
        Args:
            video_path (str): Path to the input video file
            verbose (bool): Enable verbose logging
            log_level (int): Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level, 
            format='%(asctime)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        
        self.video_path = video_path
        self.frames = []
        self.keypoints = []
        self.descriptors = []
        
    def extract_frames(self, target_fps=None):
        """
        Extract frames from the video
        
        Args:
            target_fps (float, optional): Target frames per second to sample
        """
        cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if self.verbose:
            self.logger.info(f"Video properties: {total_frames} total frames, {video_fps} fps")
        
        # Frame sampling logic
        if target_fps is not None:
            # Calculate frame interval for target fps
            frame_interval = max(1, int(video_fps / target_fps))
            if self.verbose:
                self.logger.info(f"Sampling frames at {target_fps} fps (interval: {frame_interval})")
        else:
            frame_interval = 1
        
        # Use tqdm for progress tracking if verbose is True
        frame_iterator = tqdm(range(0, total_frames, frame_interval), 
                               desc="Extracting Frames") if self.verbose else range(0, total_frames, frame_interval)
        
        for frame_idx in frame_iterator:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        
        cap.release()
        
        if self.verbose:
            self.logger.info(f"Total frames extracted: {len(self.frames)}")
        
    def detect_features(self, method='sift'):
        """
        Detect and extract features from frames
        
        Args:
            method (str): Feature detection method (sift, orb, etc.)
        """
        if self.verbose:
            self.logger.info(f"Detecting features using {method} method")
        
        if method == 'sift':
            detector = cv2.SIFT_create()
        elif method == 'orb':
            detector = cv2.ORB_create()
        else:
            raise ValueError("Unsupported feature detection method")
        
        # Use tqdm for progress tracking if verbose is True
        frame_iterator = tqdm(self.frames, desc="Detecting Features") if self.verbose else self.frames
        
        for frame in frame_iterator:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = detector.detectAndCompute(gray, None)
            
            # Convert keypoints to numpy array
            kp_np = np.array([kp.pt for kp in kp])
            
            self.keypoints.append(kp_np)
            self.descriptors.append(des)
        
        if self.verbose:
            total_keypoints = sum(len(kp) for kp in self.keypoints)
            self.logger.info(f"Total keypoints detected: {total_keypoints}")
        
    def match_features(self, matching_method='bf'):
        """
        Match features between consecutive frames
        
        Args:
            matching_method (str): Feature matching method
        """
        if self.verbose:
            self.logger.info(f"Matching features using {matching_method} method")
        
        matches_list = []
        
        if matching_method == 'bf':
            matcher = cv2.BFMatcher()
        elif matching_method == 'flann':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Use tqdm for progress tracking if verbose is True
        frame_range = tqdm(range(len(self.descriptors) - 1), desc="Matching Features") if self.verbose else range(len(self.descriptors) - 1)
        
        for i in frame_range:
            matches = matcher.match(self.descriptors[i], self.descriptors[i+1])
            matches_list.append(matches)
        
        if self.verbose:
            self.logger.info(f"Total match groups: {len(matches_list)}")
        
        return matches_list
    
    def estimate_motion(self, matches_list):
        """
        Estimate camera motion between consecutive frames
        
        Args:
            matches_list (list): List of feature matches
        
        Returns:
            list: Camera motion transformations
        """
        if self.verbose:
            self.logger.info("Estimating camera motion")
        
        camera_motions = []
        
        # Use tqdm for progress tracking if verbose is True
        matches_iterator = tqdm(enumerate(matches_list), 
                                total=len(matches_list), 
                                desc="Estimating Motion") if self.verbose else enumerate(matches_list)
        
        for i, matches in matches_iterator:
            # Get matched keypoints
            pts1 = np.float32([self.keypoints[i][m.queryIdx] for m in matches])
            pts2 = np.float32([self.keypoints[i+1][m.trainIdx] for m in matches])
            
            # Estimate Essential Matrix
            E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            # Recover pose
            _, R, t, _ = cv2.recoverPose(E, pts1, pts2)
            
            camera_motions.append((R, t))
        
        return camera_motions
    
    def triangulate_points(self, camera_motions):
        """
        Triangulate 3D points from matched features
        
        Args:
            camera_motions (list): Camera motion transformations
        
        Returns:
            numpy.ndarray: 3D point cloud
        """
        if self.verbose:
            self.logger.info("Triangulating 3D points")
        
        # Assume initial camera matrix as identity
        P0 = np.eye(3, 4)
        
        point_cloud = []
        
        # Use tqdm for progress tracking if verbose is True
        motion_iterator = tqdm(enumerate(camera_motions), 
                               total=len(camera_motions), 
                               desc="Triangulating Points") if self.verbose else enumerate(camera_motions)
        
        for i, (R, t) in motion_iterator:
            # Construct camera matrix for current frame
            P1 = np.hstack((R, t))
            
            # Get matched points for current and next frame
            pts1 = np.float32([self.keypoints[i][m.queryIdx] for m in self.matches_list[i]])
            pts2 = np.float32([self.keypoints[i+1][m.trainIdx] for m in self.matches_list[i]])
            
            # Triangulate points
            points_4d = cv2.triangulatePoints(P0, P1, pts1.T, pts2.T)
            points_3d = points_4d[:3] / points_4d[3]
            
            point_cloud.extend(points_3d.T)
        
        points_array = np.array(point_cloud)
        
        if self.verbose:
            self.logger.info(f"Total 3D points generated: {len(points_array)}")
        
        return points_array
    
    def generate_pointcloud(self, target_fps=None):
        """
        Main method to generate sparse point cloud
        
        Args:
            target_fps (float, optional): Target frames per second to sample
        
        Returns:
            open3d.geometry.PointCloud: Sparse point cloud
        """
        if self.verbose:
            self.logger.info("Starting sparse point cloud generation")
        
        # Extract frames
        self.extract_frames(target_fps)
        
        # Detect features
        self.detect_features(method='sift')
        
        # Match features
        self.matches_list = self.match_features(matching_method='bf')
        
        # Estimate camera motion
        camera_motions = self.estimate_motion(self.matches_list)
        
        # Triangulate points
        points = self.triangulate_points(camera_motions)
        
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if self.verbose:
            self.logger.info("Sparse point cloud generation complete")
        
        return pcd

# Example usage
def main():
    video_path = 'test.mp4'
    
    # Create generator with verbose logging
    sfm_generator = SparsePointCloudGenerator(video_path, verbose=True)
    
    # Generate point cloud, sampling at 2 fps
    point_cloud = sfm_generator.generate_pointcloud(target_fps=2)
    
    # Visualize point cloud
    o3d.visualization.draw_geometries([point_cloud])
    o3d.io.write_point_cloud("output_point_cloud.ply", point_cloud)
    print("Point cloud saved as output_point_cloud.ply")

if __name__ == '__main__':
    main()