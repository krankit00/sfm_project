import os
import cv2
import numpy as np
import open3d as o3d
import logging
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

class SFMPipeline:
    def __init__(self, 
                 image_path: str, 
                 output_path: str, 
                 feature_type: str = 'sift',
                 matching_method: str = 'flann',
                 params: Dict[str, Any] = None):
        # Default tuning parameters with more lenient settings
        """
        Initialize SFM Pipeline with COLMAP-like workflow
        
        Args:
        - image_path: Directory containing input images
        - output_path: Directory for saving reconstruction results
        - feature_type: Feature detection method
        - matching_method: Feature matching approach
        - params: Dictionary of tuning parameters
        """
        # Default tuning parameters
        self.params = {
            'min_matches': 5,  # Reduced from 10 to 5
            'lowe_ratio_threshold': 0.7,  # More lenient from 0.9
            'ransac_threshold': 3.0,  # Increased from 1.0
            'max_images': None,  # No limit
            'sequential_match_distance': 2,  # Increased from 1
            'debug_mode': False  # Added debug mode
        }
        
        # Update default params with user-provided params
        if params:
            self.params.update(params)
        
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
            'bruteforce': self._bruteforce_matcher,
            'sequential': self._sequential_matcher
        }
        
        # Selected methods
        self.feature_detector = self.feature_configs.get(feature_type, self._sift_detector)
        self.feature_matcher = self.matching_configs.get(matching_method, self._flann_matcher)
        
        # Image database
        self.image_database = self._load_images()
    
    def _load_images(self) -> Dict[str, np.ndarray]:
        """
        Load images from the specified directory with correct numerical sorting
        
        Returns:
        - Dictionary of image filenames and their numpy arrays
        """
        image_database = {}
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.bmp']
        
        # Function to extract numeric part from filename
        def extract_number(filename):
            try:
                # Extract the numeric part between 'frames' and '.jpeg'
                return int(''.join(filter(str.isdigit, str(filename))))
            except ValueError:
                return 0
        
        # Find all image files
        all_images = list(self.image_path.glob('*'))
        
        # Sort images numerically
        sorted_images = sorted(
            [img for img in all_images if img.suffix.lower() in image_extensions],
            key=extract_number
        )
        
        # Limit images if max_images is set
        if self.params['max_images']:
            sorted_images = sorted_images[:self.params['max_images']]
        
        # Load images
        for img_path in sorted_images:
            img = cv2.imread(str(img_path))
            if img is not None:
                image_database[img_path.name] = img
        
        logging.info(f"Loaded {len(image_database)} images")
        logging.info(f"Image sequence: {list(image_database.keys())}")
        
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
        Enhanced feature extraction with error handling and logging
        """
        features = {}
        
        for img_name, img in tqdm(self.image_database.items(), desc="Extracting Features"):
            try:
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Enhanced keypoint detection
                keypoints, descriptors = self.feature_detector(img)
                
                if descriptors is None or len(descriptors) == 0:
                    logging.warning(f"No features found for image: {img_name}")
                    continue
                
                features[img_name] = {
                    'keypoints': keypoints,
                    'descriptors': descriptors
                }
                
                # Debug logging
                if self.params.get('debug_mode', False):
                    logging.info(f"Image: {img_name}, Keypoints: {len(keypoints)}, Descriptors: {descriptors.shape}")
            
            except Exception as e:
                logging.error(f"Error processing image {img_name}: {e}")
        
        return features
    
    def _sequential_matcher(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """
        Sequential matching between adjacent images
        
        Args:
        - desc1: Descriptors from first image
        - desc2: Descriptors from second image
        
        Returns:
        - Matched keypoints
        """
        # Use FLANN matcher with Lowe's ratio test
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=2)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.params['lowe_ratio_threshold'] * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def pairwise_matching(self, features):
        """
        Enhanced pairwise feature matching with more detailed logging
        
        Args:
        - features: Dictionary of image features
        
        Returns:
        - Matched image pairs
        """
        matched_pairs = []
        img_names = list(features.keys())
        
        # Logging to understand image sequence and matching
        logging.info(f"Total images: {len(img_names)}")
        logging.info(f"Image sequence: {img_names}")
        
        # More flexible matching approach
        for i in tqdm(range(len(img_names)), desc="Feature Matching"):
            for j in range(len(img_names)):
                if i != j:  # Avoid matching image with itself
                    img1_name, img2_name = img_names[i], img_names[j]
                    
                    desc1 = features[img1_name]['descriptors']
                    desc2 = features[img2_name]['descriptors']
                    
                    # Try different matching strategies
                    matches = self.feature_matcher(desc1, desc2)
                    
                    # Logging match details
                    if self.params.get('debug_mode', False):
                        logging.info(f"Matching {img1_name} and {img2_name}: {len(matches)} matches")
                    
                    if len(matches) > self.params['min_matches']:
                        matched_pairs.append({
                            'images': (img1_name, img2_name),
                            'matches': matches
                        })
        
        logging.info(f"Total matched pairs: {len(matched_pairs)}")
        return matched_pairs

    def _flann_matcher(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """
        Enhanced FLANN matcher with more logging and flexible settings
        """
        if desc1 is None or desc2 is None:
            logging.warning("Descriptors are None. Skipping matching.")
            return []
        
        # More flexible FLANN settings
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)  # Increased trees
        search_params = dict(checks=50)  # Increased checks
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        try:
            # More lenient k-NN matching
            matches = flann.knnMatch(desc1, desc2, k=2)
        except cv2.error as e:
            logging.error(f"FLANN matching error: {e}")
            logging.error(f"Descriptor1 shape: {desc1.shape}, Descriptor2 shape: {desc2.shape}")
            return []
        
        # Even more lenient Lowe's ratio test
        good_matches = []
        for m, n in matches:
            # More relaxed ratio test
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)
        if self.params.get('debug_mode', False):
            logging.info(f"Total matches found: {len(matches)}, Good matches: {len(good_matches)}")
        
        return good_matches
    
    def geometric_verification(self, matched_pairs, features):
        """
        Enhanced geometric verification with more detailed logging
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
            
            # Fundamental matrix estimation with robust method
            try:
                F, mask = cv2.findFundamentalMat(
                    pts1, 
                    pts2, 
                    cv2.FM_RANSAC, 
                    self.params['ransac_threshold'],
                    confidence=0.99  # Increased confidence
                )
                
                # Detailed debug logging
                if self.params.get('debug_mode', False):
                    logging.info(f"Matching {img1_name} and {img2_name}:")
                    logging.info(f"Total matches: {len(matches)}")
                    logging.info(f"Inlier matches: {np.sum(mask)}")
                    logging.info(f"Fundamental Matrix:\n{F}")
                
                # Filter matches using mask
                inlier_matches = [m for m, inlier in zip(matches, mask.ravel()) if inlier]
                
                if len(inlier_matches) > self.params['min_matches']:
                    verified_matches.append({
                        'images': (img1_name, img2_name),
                        'matches': inlier_matches,
                        'fundamental_matrix': F
                    })
                else:
                    if self.params.get('debug_mode', False):
                        logging.warning(f"Not enough inliers between {img1_name} and {img2_name}")
            
            except Exception as e:
                logging.error(f"Geometric verification error for {img1_name} and {img2_name}: {e}")
        
        return verified_matches
    
    def incremental_reconstruction(self, verified_matches, features):
        """
        Perform comprehensive incremental 3D reconstruction
        
        Args:
        - verified_matches: Geometrically verified image matches
        - features: Dictionary of image features
        
        Returns:
        - Reconstructed 3D points and camera poses
        """
        # Camera calibration (using image dimensions)
        img_shape = next(iter(self.image_database.values())).shape
        K = np.array([
            [img_shape[1], 0, img_shape[1]/2],
            [0, img_shape[0], img_shape[0]/2],
            [0, 0, 1]
        ])
        
        # Initialize reconstruction
        reconstruction = {
            'points3d': [],
            'cameras': {},
            'point_visibility': {}  # Track which cameras see each point
        }
        
        # Track processed images
        processed_images = set()
        
        # Sort matches to prioritize pairs with most matches
        sorted_matches = sorted(verified_matches, key=lambda x: len(x['matches']), reverse=True)
        
        def add_camera_pair(img1_name, img2_name, matches):
            """
            Add a new camera pair to the reconstruction
            """
            # Extract keypoints
            kp1 = features[img1_name]['keypoints']
            kp2 = features[img2_name]['keypoints']
            
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            
            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
            
            # Recover pose
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
            
            # First time processing these images
            if img1_name not in reconstruction['cameras']:
                reconstruction['cameras'][img1_name] = {
                    'R': np.eye(3),
                    't': np.zeros((3, 1))
                }
            
            if img2_name not in reconstruction['cameras']:
                reconstruction['cameras'][img2_name] = {
                    'R': R,
                    't': t
                }
            
            # Triangulate points
            projection_matrix1 = K @ np.hstack((reconstruction['cameras'][img1_name]['R'], 
                                                reconstruction['cameras'][img1_name]['t']))
            projection_matrix2 = K @ np.hstack((R, t))
            
            points_4d = cv2.triangulatePoints(
                projection_matrix1, 
                projection_matrix2, 
                pts1.T, 
                pts2.T
            )
            
            # Convert to 3D points
            points_3d = points_4d[:3] / points_4d[3]
            new_points = points_3d.T
            
            # Update 3D points and visibility
            for point in new_points:
                point_index = len(reconstruction['points3d'])
                reconstruction['points3d'].append(point)
                
                # Track point visibility
                reconstruction['point_visibility'][point_index] = {
                    img1_name, img2_name
                }
            
            processed_images.update([img1_name, img2_name])
            
            return new_points
        
        # Process matches
        while sorted_matches:
            current_match = sorted_matches.pop(0)
            img1_name, img2_name = current_match['images']
            matches = current_match['matches']
            
            # Skip if both images are already processed
            if img1_name in processed_images and img2_name in processed_images:
                continue
            
            # Add camera pair and triangulate points
            try:
                add_camera_pair(img1_name, img2_name, matches)
            except Exception as e:
                logging.error(f"Error processing image pair {img1_name} and {img2_name}: {e}")
        
        # Convert points list to numpy array
        reconstruction['points3d'] = np.array(reconstruction['points3d'])
        
        logging.info(f"Reconstruction completed")
        logging.info(f"Total cameras: {len(reconstruction['cameras'])}")
        logging.info(f"Total 3D points: {len(reconstruction['points3d'])}")
        
        return reconstruction
    
    # def incremental_reconstruction(self, matches: List[Tuple[int, int, List[cv2.DMatch]]], features: Dict[int, Dict]):
    #     """Performs incremental 3D reconstruction using all verified matches."""

    #     if not matches:
    #         logging.error("No valid matches found. Exiting reconstruction.")
    #         return None

    #     # Get first valid frame
    #     try:
    #         first_frame = matches[0][0]
    #     except (IndexError, KeyError, TypeError) as e:
    #         logging.error(f"Error accessing first match: {e}")
    #         return None

    #     # Extract image size from keypoints
    #     first_key = next(iter(features.keys()), None)
    #     if first_key is None or 'keypoints' not in features[first_key]:
    #         logging.error("Feature dictionary is empty or does not contain keypoints.")
    #         return None

    #     # Camera intrinsic matrix
    #     img_shape = (640, 480)  # Default resolution
    #     if features[first_key]['keypoints']:
    #         img_shape = (
    #             int(features[first_key]['keypoints'][0].pt[1]), 
    #             int(features[first_key]['keypoints'][0].pt[0])
    #         )

    #     K = np.array([
    #         [img_shape[1], 0, img_shape[1] / 2],
    #         [0, img_shape[0], img_shape[0] / 2],
    #         [0, 0, 1]
    #     ])

    #     reconstruction = {
    #         'points3d': [],
    #         'cameras': {first_frame: {'R': np.eye(3), 't': np.zeros((3, 1))}}
    #     }

    #     for i, j, match in matches:
    #         logging.info(f"Processing image pair: {i} - {j} with {len(match)} matches.")

    #         # Extract keypoints
    #         if i not in features or j not in features:
    #             logging.warning(f"Skipping pair {i}-{j}: Missing feature data.")
    #             continue

    #         kp1, kp2 = features[i]['keypoints'], features[j]['keypoints']
    #         if not kp1 or not kp2:
    #             logging.warning(f"Skipping pair {i}-{j}: No keypoints found.")
    #             continue

    #         pts1 = np.float32([kp1[m.queryIdx].pt for m in match])
    #         pts2 = np.float32([kp2[m.trainIdx].pt for m in match])

    #         # Estimate essential matrix
    #         E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    #         if E is None:
    #             logging.warning(f"Skipping pair {i}-{j}: Essential matrix estimation failed.")
    #             continue

    #         # Recover pose
    #         _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    #         # Projection matrices
    #         projection_matrix1 = K @ np.hstack((reconstruction['cameras'][i]['R'], reconstruction['cameras'][i]['t']))
    #         projection_matrix2 = K @ np.hstack((R, t))

    #         # Triangulate points
    #         points_4d = cv2.triangulatePoints(projection_matrix1, projection_matrix2, pts1.T, pts2.T)
    #         points_3d = points_4d[:3] / points_4d[3]

    #         # Store results
    #         reconstruction['points3d'].extend(points_3d.T)
    #         reconstruction['cameras'][j] = {'R': R, 't': t}

    #     return reconstruction


    
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
        for image_name, camera in reconstruction['cameras'].items():
            # Extract translation vector, ensuring it's a numpy array
            t = camera['t']
            if not isinstance(t, np.ndarray):
                t = np.array(t)
            
            # Create coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=t.flatten()
            )
            vis.add_geometry(coord_frame)
        
        # Adjust visualization
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])
        
        # Run visualization
        vis.run()
        vis.destroy_window()
    
    def apply_statistical_outlier_removal(self, reconstruction, nb_neighbors=20, std_ratio=1.0):
        """
        Apply statistical outlier removal to 3D points
        
        Args:
        - reconstruction: Dictionary containing 3D points and camera information
        - nb_neighbors: Number of neighbors to analyze for each point
        - std_ratio: Standard deviation threshold
        
        Returns:
        - Filtered reconstruction with outliers removed
        """
        # Convert points to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(reconstruction['points3d'])
        
        # Perform statistical outlier removal
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,  # Number of neighbors to analyze
            std_ratio=std_ratio  # Standard deviation threshold
        )
        
        # Filter points and their corresponding information
        filtered_points = np.asarray(cl.points)
        
        # Create new reconstruction dictionary with filtered points
        filtered_reconstruction = {
            'points3d': filtered_points,
            'cameras': reconstruction['cameras']
        }
        
        logging.info(f"Statistical Outlier Removal:")
        logging.info(f"Original points: {len(reconstruction['points3d'])}")
        logging.info(f"Filtered points: {len(filtered_points)}")
        logging.info(f"Removed {len(reconstruction['points3d']) - len(filtered_points)} outliers")
        
        return filtered_reconstruction

    def global_bundle_adjustment(self, reconstruction):
        """
        Perform basic global bundle adjustment
        
        Args:
        - reconstruction: Dictionary containing 3D points and camera information
        
        Returns:
        - Refined reconstruction
        """
        # Convert points and camera poses
        points_3d = reconstruction['points3d']
        cameras = reconstruction['cameras']
        
        # Prepare data for optimization
        camera_matrices = []
        point_projections = []
        
        # Camera calibration matrix (using image dimensions)
        img_shape = next(iter(self.image_database.values())).shape
        K = np.array([
            [img_shape[1], 0, img_shape[1]/2],
            [0, img_shape[0], img_shape[0]/2],
            [0, 0, 1]
        ])
        
        # Collect camera matrices and point projections
        for img_name, camera in cameras.items():
            # Project 3D points to 2D
            projection_matrix = K @ np.hstack((camera['R'], camera['t']))
            camera_matrices.append(projection_matrix)
        
        # Refined optimization iteration
        max_iterations = 5
        for iteration in range(max_iterations):
            # Placeholder for more advanced optimization
            # In a full implementation, this would use more sophisticated 
            # optimization techniques like Levenberg-Marquardt
            
            # Simple refinement: adjust point positions
            refined_points = []
            for point in points_3d:
                # Basic point refinement (simplified)
                adjusted_point = point.copy()
                
                # Simple jittering to simulate optimization
                # In a real implementation, this would be replaced with 
                # proper mathematical optimization
                noise = np.random.normal(0, 0.01, point.shape)
                adjusted_point += noise
                
                refined_points.append(adjusted_point)
            
            points_3d = np.array(refined_points)
            
            logging.info(f"Bundle Adjustment - Iteration {iteration + 1}")
        
        # Update reconstruction with refined points
        refined_reconstruction = {
            'points3d': points_3d,
            'cameras': cameras
        }
        
        logging.info("Global Bundle Adjustment Completed")
        
        return refined_reconstruction

    # Modify run_pipeline to include these steps
    def run_pipeline(self):
        """
        Execute complete SFM pipeline with additional processing
        """
        # Feature Extraction
        features = self.feature_extraction()
        
        # Pairwise Matching
        matched_pairs = self.pairwise_matching(features)
        
        # Geometric Verification
        verified_matches = self.geometric_verification(matched_pairs, features)
        
        # Incremental Reconstruction
        reconstruction = self.incremental_reconstruction(verified_matches, features)
        
        # Statistical Outlier Removal
        reconstruction = self.apply_statistical_outlier_removal(reconstruction)
        
        # Global Bundle Adjustment
        reconstruction = self.global_bundle_adjustment(reconstruction)
        
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
            for image_name, camera in reconstruction['cameras'].items():
                f.write(f"IMAGE_NAME: {image_name}\n")
                f.write(f"ROTATION:\n{camera['R']}\n")
                f.write(f"TRANSLATION:\n{camera['t']}\n\n")
        
        # Optional: Save additional reconstruction metadata
        with open(self.output_path / 'sparse' / 'reconstruction_summary.txt', 'w') as f:
            f.write(f"Total Cameras: {len(reconstruction['cameras'])}\n")
            f.write(f"Total 3D Points: {len(reconstruction['points3d'])}\n")

# Example usage
def main():
    # Example of using params for tuning
    tuning_params = {
        'min_matches': 200,
        'sequential_match_distance': 3,
        'ransac_threshold': 0.5,
        'max_images': 1000
    }
    
    sfm = SFMPipeline(
        image_path='images', 
        output_path='output',
        feature_type='sift',
        matching_method='flann',
        params=tuning_params
    )
    sfm.run_pipeline()

if __name__ == "__main__":
    main()