import cv2
import numpy as np
import os
import glob
from scipy.spatial import cKDTree
from pyntcloud import PyntCloud
import pandas as pd

# ===================== PARAMETERS =====================
VIDEO_PATH = "watchtower.mp4"  # Path to input video
IMAGE_DIR = "images"  # Directory of images if using pre-extracted frames
FPS = 2  # Frames per second for extraction
FEATURE_TYPE = "SIFT"  # Options: "ORB", "SIFT", "SURF"
MATCHER_TYPE = "BF"  # Options: "BF", "FLANN"
MIN_MATCH_COUNT = 20  # Minimum matches for pose estimation
FILTER_TYPE = "STATISTICAL"  # Options: "STATISTICAL", "RADIAL"
K_NEIGHBORS = 10  # Statistical filter param
STD_DEV = 2.0  # Outlier threshold for filtering

# ===================== FRAME EXTRACTION =====================
def extract_frames(video_path, output_dir, fps):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval = int(frame_rate / fps)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            cv2.imwrite(f"{output_dir}/frame_{count}.png", frame)
        count += 1
    cap.release()

def load_images(image_dir):
    images = sorted(glob.glob(f"{image_dir}/*.png"), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    return [cv2.imread(img) for img in images]

# ===================== CAMERA INTRINSICS =====================
def estimate_intrinsics(image):
    h, w = image.shape[:2]
    focal_length = max(h, w)
    K = np.array([[focal_length, 0, w / 2],
                  [0, focal_length, h / 2],
                  [0, 0, 1]], dtype=np.float32)
    return K

# ===================== FEATURE EXTRACTION =====================
def get_features(image, feature_type):
    if feature_type == "SIFT":
        detector = cv2.SIFT_create()
    elif feature_type == "ORB":
        detector = cv2.ORB_create()
    elif feature_type == "SURF":
        detector = cv2.xfeatures2d.SURF_create()
    else:
        raise ValueError("Invalid feature type")
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors

# ===================== FEATURE MATCHING =====================
def match_features(desc1, desc2, matcher_type):
    if matcher_type == "BF":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.match(desc1, desc2)
    return sorted(matches, key=lambda x: x.distance)

# ===================== FILTERING POINT CLOUD =====================
def filter_point_cloud(points, method):
    if method == "STATISTICAL":
        cloud = PyntCloud(pd.DataFrame(points, columns=["x", "y", "z"]))
        kdtree = cloud.add_structure("kdtree")
        k_neighbors = np.array([cloud.get_neighbors(kdtree, k=K_NEIGHBORS) for _ in range(len(points))])
        mean_dist = np.mean(k_neighbors, axis=1)
        threshold = np.mean(mean_dist) + STD_DEV * np.std(mean_dist)
        return points[mean_dist < threshold]
    elif method == "RADIAL":
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)
        return points[dists[:, 1] < np.mean(dists) + STD_DEV * np.std(dists)]
    return points

# ===================== MAIN PIPELINE =====================
if __name__ == "__main__":
    extract_frames(VIDEO_PATH, IMAGE_DIR, FPS)
    images = load_images(IMAGE_DIR)
    K = estimate_intrinsics(images[0])
    keypoints_all, descriptors_all = zip(*[get_features(img, FEATURE_TYPE) for img in images])
    
    # Match features between consecutive frames
    matches_all = [match_features(descriptors_all[i], descriptors_all[i+1], MATCHER_TYPE) for i in range(len(images)-1)]
    
    # Visualize feature matches
    for i in range(min(3, len(matches_all))):
        img_matches = cv2.drawMatches(images[i], keypoints_all[i], images[i+1], keypoints_all[i+1], matches_all[i], None)
        cv2.imshow("Matches", img_matches)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Sparse point cloud generation (Mocked here)
    sparse_cloud = np.random.rand(1000, 3)  # Replace with real triangulation
    np.savetxt("sparse_cloud.txt", sparse_cloud)
    
    # Filter point cloud
    filtered_cloud = filter_point_cloud(sparse_cloud, FILTER_TYPE)
    np.savetxt("filtered_cloud.txt", filtered_cloud)
