import os
import cv2
import numpy as np
import open3d as o3d
import logging
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict

match_threshold = 2

class SFMPipeline:
    def __init__(self, 
                 image_path: str, 
                 output_path: str, 
                 feature_type: str = 'sift',
                 matching_method: str = 'flann'):
        """
        Initialize SFM Pipeline with COLMAP-like workflow
        
        Args:
        - image_path: Directory containing input images
        - output_path: Directory for saving reconstruction results
        - feature_type: Feature detection method
        - matching_method: Feature matching approach
        """
        self.image_path = Path(image_path)
        self.output_path = Path(output_path)
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / 'sparse').mkdir(exist_ok=True)
        (self.output_path / 'dense').mkdir(exist_ok=True)
        
        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(self.output_path / 'sfm_log.txt'),
                logging.StreamHandler()
            ]
        )
        
        # Feature detection configurations
        self.feature_configs = {
            'sift': self._sift_detector,
            'surf': self._surf_detector,
            'orb': self._orb_detector
        }
        
        # Matching method configurations
        self.matching_configs = {
            'flann': self._flann_matcher,
            'bruteforce': self._bruteforce_matcher
        }
        
        # Selected methods
        self.feature_detector = self.feature_configs.get(feature_type, self._sift_detector)
        self.feature_matcher = self.matching_configs.get(matching_method, self._flann_matcher)
        
        # Image database
        self.image_database = self._load_images()
    
    def _load_images(self) -> Dict[str, np.ndarray]:
        """
        Load images from the specified directory
        
        Returns:
        - Dictionary of image filenames and their numpy arrays
        """
        image_database = {}
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.bmp']
        
        for img_path in sorted(self.image_path.glob('*'), key=lambda s: int(str(s)[-9:-6])):
            if img_path.suffix.lower() in image_extensions:
                img = cv2.imread(str(img_path))
                if img is not None:
                    image_database[img_path.name] = img
        logging.info(f"Loaded {len(image_database)} images")
        return image_database
    
    def _sift_detector(self, image: np.ndarray) -> Tuple:
        """
        SIFT feature detection
        
        Args:
        - image: Input image
        
        Returns:
        - Keypoints
        - Descriptors
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def _surf_detector(self, image: np.ndarray) -> Tuple:
        """
        SURF feature detection
        
        Args:
        - image: Input image
        
        Returns:
        - Keypoints
        - Descriptors
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        surf = cv2.xfeatures2d.SURF_create()
        keypoints, descriptors = surf.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def _orb_detector(self, image: np.ndarray) -> Tuple:
        """
        ORB feature detection
        
        Args:
        - image: Input image
        
        Returns:
        - Keypoints
        - Descriptors
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def _flann_matcher(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """
        FLANN-based feature matching
        
        Args:
        - desc1: Descriptors from first image
        - desc2: Descriptors from second image
        
        Returns:
        - Matched keypoints
        """
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=2)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def _bruteforce_matcher(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """
        Brute-force feature matching
        
        Args:
        - desc1: Descriptors from first image
        - desc2: Descriptors from second image
        
        Returns:
        - Matched keypoints
        """
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    
    def feature_extraction(self):
        """
        Extract features from all images
        
        Returns:
        - Dictionary of image features
        """
        features = {}
        
        for img_name, img in tqdm(self.image_database.items(), desc="Extracting Features"):
            keypoints, descriptors = self.feature_detector(img)
            features[img_name] = {
                'keypoints': keypoints,
                'descriptors': descriptors
            }
        
        return features
    
    def pairwise_matching(self, features):
        """
        Perform pairwise feature matching
        
        Args:
        - features: Dictionary of image features
        
        Returns:
        - Matched image pairs
        """
        matched_pairs = []
        img_names = list(features.keys())
        
        for i in tqdm(range(len(img_names)), desc="Feature Matching"):
            for j in range(i+1, len(img_names)):
                img1_name, img2_name = img_names[i], img_names[j]
                
                desc1 = features[img1_name]['descriptors']
                desc2 = features[img2_name]['descriptors']
                
                matches = self.feature_matcher(desc1, desc2)
                
                if len(matches) > match_threshold:  # Minimum match threshold
                    matched_pairs.append({
                        'images': (img1_name, img2_name),
                        'matches': matches
                    })
        
        return matched_pairs
    
    def geometric_verification(self, matched_pairs, features):
        """
        Geometrically verify matched image pairs
        
        Args:
        - matched_pairs: List of matched image pairs
        - features: Dictionary of image features
        
        Returns:
        - Verified image matches
        """
        verified_matches = []
        
        for pair in tqdm(matched_pairs, desc="Geometric Verification"):
            img1_name, img2_name = pair['images']
            matches = pair['matches']
            
            kp1 = features[img1_name]['keypoints']
            kp2 = features[img2_name]['keypoints']
            
            # Extract matched keypoints
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            
            # Fundamental matrix estimation
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
            print(F)
            
            # Filter matches using mask
            inlier_matches = [m for m, inlier in zip(matches, mask.ravel()) if inlier]
            
            if len(inlier_matches) > match_threshold:
                verified_matches.append({
                    'images': (img1_name, img2_name),
                    'matches': inlier_matches,
                    'fundamental_matrix': F
                })
        
        return verified_matches
    
    def incremental_reconstruction(self, verified_matches, features):
        """
        Perform incremental 3D reconstruction
        
        Args:
        - verified_matches: Geometrically verified image matches
        - features: Dictionary of image features
        
        Returns:
        - Reconstructed 3D points and camera poses
        """
        # Camera calibration (simplified)
        img_shape = next(iter(self.image_database.values())).shape
        K = np.array([
            [img_shape[1], 0, img_shape[1]/2],
            [0, img_shape[0], img_shape[0]/2],
            [0, 0, 1]
        ])
        
        # Initialize reconstruction
        reconstruction = {
            'points3d': [],
            'cameras': []
        }
        
        # Select initial image pair
        initial_pair = verified_matches[0]
        img1_name, img2_name = initial_pair['images']
        matches = initial_pair['matches']
        
        # Extract keypoints for initial pair
        kp1 = features[img1_name]['keypoints']
        kp2 = features[img2_name]['keypoints']
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, K)
        
        # Recover pose
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
        
        # Triangulate points
        projection_matrix1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        projection_matrix2 = K @ np.hstack((R, t))
        
        points_4d = cv2.triangulatePoints(
            projection_matrix1, 
            projection_matrix2, 
            pts1.T, 
            pts2.T
        )
        
        # Convert to 3D points
        points_3d = points_4d[:3] / points_4d[3]
        
        reconstruction['points3d'] = points_3d.T
        reconstruction['cameras'].append({
            'image': img1_name,
            'R': np.eye(3),
            't': np.zeros((3, 1))
        })
        reconstruction['cameras'].append({
            'image': img2_name,
            'R': R,
            't': t
        })
        
        return reconstruction
    
    def visualize_reconstruction(self, reconstruction):
        """
        Visualize 3D reconstruction using Open3D
        
        Args:
        - reconstruction: Reconstructed 3D points and camera poses
        """
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(reconstruction['points3d'])
        
        # Visualize
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        
        # Add camera coordinate frames
        for camera in reconstruction['cameras']:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=camera['t'].flatten()
            )
            vis.add_geometry(coord_frame)
        
        vis.run()
        vis.destroy_window()
    
    def run_pipeline(self):
        """
        Execute complete SFM pipeline
        """
        # Feature Extraction
        features = self.feature_extraction()
        
        # Pairwise Matching
        matched_pairs = self.pairwise_matching(features)
        
        # Geometric Verification
        verified_matches = self.geometric_verification(matched_pairs, features)
        
        # Incremental Reconstruction
        reconstruction = self.incremental_reconstruction(verified_matches, features)
        
        # Save results
        self._save_reconstruction(reconstruction)
        # Visualization

        self.visualize_reconstruction(reconstruction)
        
    
    def _save_reconstruction(self, reconstruction):
        """
        Save reconstruction results
        
        Args:
        - reconstruction: Reconstructed 3D points and camera poses
        """
        # Save point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(reconstruction['points3d'])
        o3d.io.write_point_cloud(
            str(self.output_path / 'sparse' / 'point_cloud.ply'), 
            pcd
        )
        
        # Save camera poses
        with open(self.output_path / 'sparse' / 'cameras.txt', 'w') as f:
            for camera in reconstruction['cameras']:
                f.write(f"IMAGE_NAME: {camera['image']}\n")
                f.write(f"ROTATION:\n{camera['R']}\n")
                f.write(f"TRANSLATION:\n{camera['t']}\n\n")

# Example usage
def main():
    sfm = SFMPipeline(
        image_path='images', 
        output_path='output',
        feature_type='orb  ',
        matching_method='flann'
    )
    sfm.run_pipeline()

if __name__ == "__main__":
    main()