import pathlib
from sfm_colmap import run_sfm 
from utils import extract_frames, process_pointcloud
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run COLMAP Structure-from-Motion pipeline using PyCOLMAP.")
    
    parser.add_argument("--video_path", type=str, help="Path to the input video file.", default="../watchtower.mp4")
    parser.add_argument("--image_dir", type=str, help="Directory to save extracted frames.", default="extracted_frames")
    parser.add_argument("--output_path", type=str, help="Path to store SfM results.", default="output")
    parser.add_argument("--fps", type=int, help="Frames per second for extraction.", default=1)

    args = parser.parse_args()

    image_dir = pathlib.Path(args.image_dir)
    output_path = pathlib.Path(args.output_path)
    extract_frames(args.video_path, image_dir, fps=args.fps)
    maps = run_sfm(image_dir=image_dir, output_path=output_path)
    process_pointcloud(output_path, view_pcd=True, save_pcd=True, filter_pcd=True, is_dense=True)

if __name__ == "__main__":
    main()
