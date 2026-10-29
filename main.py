"""
Main Pipeline Runner for Computer Vision Project.

Provides a unified command-line interface to:
  --validate : Inspect environment dependencies and personal input media.
  --task 1   : Image Restoration and Segmentation.
  --task 2   : Feature Extraction and Image Matching (HOG, SIFT, Invariance).
  --task 3   : Object Recognition (Haar Cascade vs. Ultralytics YOLO).
  --task 4   : Video Motion Estimation (Differencing, MOG2, Lucas-Kanade).
  --all      : Run the complete assignment end-to-end.
"""

import sys
import os
import argparse
import time

from src.validators import validate_all_inputs, InputValidationError
from src.utils import print_banner


def main():
    parser = argparse.ArgumentParser(
        description="Computer Vision Individual Assignment - Complete Practical Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python main.py --validate             # Check input media and package status
  python main.py --task 1               # Run Task 1 (Restoration & Segmentation)
  python main.py --task 2               # Run Task 2 (Features & SIFT Matching)
  python main.py --task 3               # Run Task 3 (Haar & YOLO Object Recognition)
  python main.py --task 4               # Run Task 4 (Video Motion Estimation)
  python main.py --all                  # Run all tasks sequentially
        """
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate that all required input media files exist and are readable."
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run a specific task (1, 2, 3, or 4)."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all four tasks sequentially."
    )
    parser.add_argument(
        "--photo1",
        default="inputs/photo_original.jpg",
        help="Path to primary photograph (default: inputs/photo_original.jpg)"
    )
    parser.add_argument(
        "--photo2",
        default="inputs/photo_second.jpg",
        help="Path to second photograph for matching (default: inputs/photo_second.jpg)"
    )
    parser.add_argument(
        "--video",
        default="inputs/video_original.mp4",
        help="Path to motion video clip (default: inputs/video_original.mp4)"
    )
    parser.add_argument(
        "--output-base",
        default="outputs",
        help="Base directory for generated outputs (default: outputs)"
    )
    
    args = parser.parse_args()
    
    # If no flags provided, display interactive help and status check
    if not (args.validate or args.task or args.all):
        print("\n" + "=" * 76)
        print(" COMPUTER VISION PRACTICAL ASSIGNMENT PIPELINE")
        print("=" * 76)
        print(" No task flag specified. Running input media validation check...\n")
        validate_all_inputs(verbose=True)
        print(" To run individual tasks, use:")
        print("   python main.py --task 1   # Image Restoration & Segmentation")
        print("   python main.py --task 2   # Feature Extraction & Matching")
        print("   python main.py --task 3   # Object Recognition (Haar & YOLO)")
        print("   python main.py --task 4   # Video Motion Estimation")
        print("   python main.py --all      # Run All Tasks Sequentially")
        print("=" * 76 + "\n")
        sys.exit(0)
        
    # Validation flag
    if args.validate:
        all_ok = validate_all_inputs(verbose=True)
        sys.exit(0 if all_ok else 1)
        
    tasks_to_run = []
    if args.all:
        tasks_to_run = [1, 2, 3, 4]
    elif args.task:
        tasks_to_run = [args.task]
        
    print_banner(
        "STARTING COMPUTER VISION PIPELINE",
        f"Tasks to execute: {tasks_to_run}"
    )
    
    start_total = time.time()
    
    for task_id in tasks_to_run:
        try:
            if task_id == 1:
                from src.task1_restoration_segmentation import run_task1
                out_dir = os.path.join(args.output_base, "task1")
                run_task1(photo_path=args.photo1, output_dir=out_dir)
                
            elif task_id == 2:
                from src.task2_features_matching import run_feature_matching_pipeline
                out_dir = os.path.join(args.output_base, "task2")
                run_feature_matching_pipeline(
                    photo1_path=args.photo1,
                    photo2_path=args.photo2,
                    output_dir=out_dir
                )
                
            elif task_id == 3:
                from src.task3_object_recognition import run_task3
                out_dir = os.path.join(args.output_base, "task3")
                run_task3(photo_path=args.photo1, output_dir=out_dir)
                
            elif task_id == 4:
                from src.task4_motion_estimation import run_motion_estimation_pipeline
                out_dir = os.path.join(args.output_base, "task4")
                run_motion_estimation_pipeline(video_path=args.video, output_dir=out_dir)
                
        except InputValidationError as e:
            print(e)
            print(f"[HALTED] Task {task_id} aborted due to missing input media.")
            print("Please provide the required file and run the command again.\n")
            sys.exit(1)
        except Exception as e:
            print(f"\n[UNEXPECTED ERROR in Task {task_id}]: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    total_elapsed = time.time() - start_total
    print("\n" + "=" * 76)
    print(f" ALL REQUESTED TASKS COMPLETED SUCCESSFULLY in {total_elapsed:.2f} seconds!")
    print(f" View your generated figures, tables, and metrics inside '{args.output_base}/'.")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
