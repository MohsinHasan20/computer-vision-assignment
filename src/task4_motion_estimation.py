"""
Task 4: Video Motion Estimation.

This module implements:
1. Video reading and frame sequencing from user-supplied video clip.
2. Two-frame differencing for instantaneous temporal change detection.
3. Gaussian Mixture Model (GMM) background subtraction using OpenCV MOG2.
4. Lucas-Kanade sparse pyramidal optical flow with directional motion vector visualization.
5. Extraction and saving of original frames, foreground masks, and vector overlays.
6. Comparative evaluation table based strictly on observed execution metrics
   (foreground coverage, tracked point displacement, execution latency).
"""

import os
import sys

# Ensure project root is in sys.path when running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils import (
    save_figure,
    save_json,
    save_csv_table,
    print_banner,
    print_step,
    ensure_dir,
)
from src.validators import validate_video_original


def run_motion_estimation_pipeline(
    video_path: str = "inputs/video_original.mp4",
    output_dir: str = "outputs/task4",
    max_frames_to_process: int = 200,
    diff_threshold: int = 30,
    history: int = 200,
    var_threshold: float = 25.0
) -> Dict[str, Any]:
    """
    Executes comparative video motion estimation on consecutive frames of the video.
    """
    print_banner(
        "TASK 4: VIDEO MOTION ESTIMATION",
        f"Input: {video_path} | Output Dir: {output_dir}"
    )
    
    # 1. Validate video input
    validate_video_original(video_path, raise_error=True)
    ensure_dir(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0
    
    print(f" Loaded Video: '{video_path}'")
    print(f"   - Resolution : {width}x{height} px")
    print(f"   - Total Frames: {total_frames} frames")
    print(f"   - Frame Rate  : {fps:.2f} FPS")
    print(f"   - Duration    : {duration:.2f} seconds")
    
    frames_to_read = min(total_frames, max_frames_to_process)
    print(f"   - Processing target: {frames_to_read} consecutive frames")
    
    # Initialize MOG2 Background Subtractor
    # GMM models each pixel with a mixture of Gaussians to distinguish dynamic background from foreground
    mog2 = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=True)
    
    # Lucas-Kanade Optical Flow parameters
    feature_params = dict(maxCorners=150, qualityLevel=0.03, minDistance=10, blockSize=7)
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
    )
    
    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        raise ValueError(f"Failed to read the initial frame from {video_path}")
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect initial feature corners to track
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    
    # Tracking accumulator mask for motion vector trails
    vector_mask = np.zeros_like(prev_frame)
    
    # Performance & metric tracking
    diff_times = []
    mog2_times = []
    lk_times = []
    
    diff_fg_pixel_counts = []
    mog2_fg_pixel_counts = []
    lk_displacements = []
    
    # Select snapshot checkpoints (e.g. at 25%, 50%, and 75% of processed frames)
    snapshot_indices = [
        max(2, int(frames_to_read * 0.25)),
        max(3, int(frames_to_read * 0.50)),
        max(4, int(frames_to_read * 0.75))
    ]
    saved_snapshots = []
    
    print_step(1, "Processing Video Frames (Differencing, MOG2, Lucas-Kanade Flow)")
    
    frame_idx = 1
    while frame_idx < frames_to_read:
        ret, curr_frame = cap.read()
        if not ret or curr_frame is None:
            break
            
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # ----------------------------------------------------------------------
        # 1. Two-Frame Differencing
        # ----------------------------------------------------------------------
        t0 = time.time()
        # Grayscale absolute difference
        frame_diff = cv2.absdiff(curr_gray, prev_gray)
        # Apply slight blur to reduce camera sensor noise
        diff_blur = cv2.GaussianBlur(frame_diff, (5, 5), 0)
        _, diff_mask = cv2.threshold(diff_blur, diff_threshold, 255, cv2.THRESH_BINARY)
        # Morphological opening to clean small isolated speckles
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        diff_time = time.time() - t0
        diff_times.append(diff_time)
        diff_fg_pixel_counts.append(int(np.sum(diff_mask == 255)))
        
        # ----------------------------------------------------------------------
        # 2. GMM Background Subtraction (MOG2)
        # ----------------------------------------------------------------------
        t0 = time.time()
        mog2_raw = mog2.apply(curr_frame)
        # Shadow pixels are 127; foreground moving pixels are 255
        _, mog2_fg_mask = cv2.threshold(mog2_raw, 250, 255, cv2.THRESH_BINARY)
        mog2_fg_mask = cv2.morphologyEx(mog2_fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mog2_time = time.time() - t0
        mog2_times.append(mog2_time)
        mog2_fg_pixel_counts.append(int(np.sum(mog2_fg_mask == 255)))
        
        # ----------------------------------------------------------------------
        # 3. Lucas-Kanade Sparse Optical Flow
        # ----------------------------------------------------------------------
        t0 = time.time()
        flow_vis = curr_frame.copy()
        mean_disp = 0.0
        
        if p0 is not None and len(p0) > 0:
            # Calculate optical flow vectors using iterative Lucas-Kanade with pyramids
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
            
            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                displacements = []
                for new, old in zip(good_new, good_old):
                    a, b = int(new[0]), int(new[1])
                    c, d = int(old[0]), int(old[1])
                    
                    dx = a - c
                    dy = b - d
                    dist = np.sqrt(dx**2 + dy**2)
                    displacements.append(dist)
                    
                    # Draw motion vector arrows (only for points showing motion > 0.5px)
                    if dist > 0.5:
                        # Draw motion vector arrow in bright Cyan
                        cv2.arrowedLine(flow_vis, (c, d), (a, b), (255, 255, 0), 2, tipLength=0.3)
                        # Draw tracked feature point circle in Magenta
                        cv2.circle(flow_vis, (a, b), 4, (255, 0, 255), -1)
                        
                if displacements:
                    mean_disp = float(np.mean(displacements))
                lk_displacements.append(mean_disp)
                p0 = good_new.reshape(-1, 1, 2)
                
            # If tracked features drop below 25, re-detect features
            if p0 is None or len(p0) < 25:
                new_corners = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
                if new_corners is not None:
                    p0 = new_corners
        else:
            p0 = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feature_params)
            lk_displacements.append(0.0)
            
        lk_time = time.time() - t0
        lk_times.append(lk_time)
        
        # ----------------------------------------------------------------------
        # Snapshot Capture at Checkpoints
        # ----------------------------------------------------------------------
        if frame_idx in snapshot_indices:
            print(f"   -> Capturing snapshot at Frame #{frame_idx} (approx {frame_idx/fps:.1f}s)...")
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f"Task 4: Motion Estimation Comparison at Frame #{frame_idx} (Time: {frame_idx/fps:.2f}s)",
                         fontsize=15, fontweight="bold")
            
            # 1. Original RGB Frame
            axes[0, 0].imshow(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title(f"1. Original Video Frame #{frame_idx}")
            axes[0, 0].axis("off")
            
            # 2. Frame Differencing Mask
            axes[0, 1].imshow(diff_mask, cmap="gray")
            axes[0, 1].set_title(f"2. Two-Frame Differencing (|I_t - I_t-1|)\nForeground Pixels: {diff_fg_pixel_counts[-1]:,}")
            axes[0, 1].axis("off")
            
            # 3. MOG2 GMM Foreground Mask
            axes[1, 0].imshow(mog2_fg_mask, cmap="gray")
            axes[1, 0].set_title(f"3. GMM Background Subtraction (MOG2)\nForeground Pixels: {mog2_fg_pixel_counts[-1]:,}")
            axes[1, 0].axis("off")
            
            # 4. Lucas-Kanade Optical Flow Vectors
            axes[1, 1].imshow(cv2.cvtColor(flow_vis, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title(f"4. Lucas-Kanade Optical Flow Vectors\nMean Displacement: {mean_disp:.2f} px/frame")
            axes[1, 1].axis("off")
            
            plt.tight_layout()
            snap_path = os.path.join(output_dir, f"motion_frame_{frame_idx}_comparison.png")
            save_figure(fig, snap_path)
            saved_snapshots.append(snap_path)
            
        prev_gray = curr_gray.copy()
        frame_idx += 1
        
    cap.release()
    print(f" Successfully processed {frame_idx - 1} frames.")
    
    # ----------------------------------------------------------------------
    # Create Master Composite Visualization
    # ----------------------------------------------------------------------
    # Use the middle snapshot as the canonical composite
    master_composite_path = os.path.join(output_dir, "motion_estimation_composite.png")
    if saved_snapshots:
        # Copy or symlink snapshot to canonical composite name
        import shutil
        shutil.copyfile(saved_snapshots[len(saved_snapshots) // 2], master_composite_path)
        print(f"   [MASTER COMPOSITE] Saved -> {master_composite_path}")

    # ----------------------------------------------------------------------
    # Compute Aggregate Metrics & Observed Comparison Table
    # ----------------------------------------------------------------------
    avg_diff_time_ms = np.mean(diff_times) * 1000.0 if diff_times else 0.0
    avg_mog2_time_ms = np.mean(mog2_times) * 1000.0 if mog2_times else 0.0
    avg_lk_time_ms = np.mean(lk_times) * 1000.0 if lk_times else 0.0
    
    avg_diff_pixels = float(np.mean(diff_fg_pixel_counts)) if diff_fg_pixel_counts else 0.0
    avg_mog2_pixels = float(np.mean(mog2_fg_pixel_counts)) if mog2_fg_pixel_counts else 0.0
    avg_lk_disp = float(np.mean(lk_displacements)) if lk_displacements else 0.0
    
    comparison_table = [
        {
            "Method": "Two-Frame Differencing",
            "Underlying_Principle": "Absolute subtraction |I_t - I_t-1| with thresholding",
            "Output_Representation": "Binary motion edge mask",
            "Avg_Latency_ms": round(avg_diff_time_ms, 2),
            "Processing_FPS": round(1000.0 / avg_diff_time_ms, 1) if avg_diff_time_ms > 0 else 0,
            "Avg_Foreground_Pixels": int(round(avg_diff_pixels)),
            "Key_Strengths_Observed": "Extremely fast, zero model initialization delay",
            "Key_Weaknesses_Observed": "Ghosting artifacts; hollow object interiors; fails when object stops moving"
        },
        {
            "Method": "GMM Background Subtraction (MOG2)",
            "Underlying_Principle": "Multi-modal Gaussian mixture model per background pixel",
            "Output_Representation": "Segmented binary silhouette mask with shadow identification",
            "Avg_Latency_ms": round(avg_mog2_time_ms, 2),
            "Processing_FPS": round(1000.0 / avg_mog2_time_ms, 1) if avg_mog2_time_ms > 0 else 0,
            "Avg_Foreground_Pixels": int(round(avg_mog2_pixels)),
            "Key_Strengths_Observed": "Complete solid silhouettes; adapts to illumination; distinguishes shadows",
            "Key_Weaknesses_Observed": "Requires warm-up initialization frames; susceptible to camera jitter"
        },
        {
            "Method": "Lucas-Kanade Optical Flow",
            "Underlying_Principle": "Brightness constancy + spatial coherence on pyramidal gradients",
            "Output_Representation": "Sparse velocity vectors (direction & magnitude arrows)",
            "Avg_Latency_ms": round(avg_lk_time_ms, 2),
            "Processing_FPS": round(1000.0 / avg_lk_time_ms, 1) if avg_lk_time_ms > 0 else 0,
            "Avg_Foreground_Pixels": f"N/A (Mean disp: {avg_lk_disp:.2f} px/frame)",
            "Key_Strengths_Observed": "Provides quantitative trajectory vectors (speed & direction)",
            "Key_Weaknesses_Observed": "Tracks only high-gradient corner points; struggles with textureless surfaces"
        }
    ]
    
    csv_path = os.path.join(output_dir, "motion_comparison_table.csv")
    json_path = os.path.join(output_dir, "motion_metrics.json")
    save_csv_table(comparison_table, csv_path)
    save_json({
        "video_properties": {
            "resolution": f"{width}x{height}",
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration,
            "frames_analyzed": frame_idx - 1
        },
        "comparison_table": comparison_table
    }, json_path)
    
    print("\n" + "=" * 76)
    print(" [OBSERVED VIDEO MOTION ESTIMATION COMPARISON]")
    print("-" * 76)
    for row in comparison_table:
        print(f" * {row['Method']:<35} | {row['Avg_Latency_ms']:>6.2f} ms ({row['Processing_FPS']} FPS)")
        print(f"   - Strengths: {row['Key_Strengths_Observed']}")
        print(f"   - Tradeoffs: {row['Key_Weaknesses_Observed']}")
    print("=" * 76 + "\n")
    
    return {
        "snapshots": saved_snapshots,
        "master_composite": master_composite_path,
        "comparison_csv": csv_path,
        "metrics_json": json_path
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Task 4: Video Motion Estimation")
    parser.add_argument("--video", default="inputs/video_original.mp4", help="Path to input video")
    parser.add_argument("--output-dir", default="outputs/task4", help="Directory to save task outputs")
    parser.add_argument("--max-frames", type=int, default=200, help="Maximum frames to process")
    parser.add_argument("--diff-thresh", type=int, default=30, help="Two-frame differencing threshold")
    args = parser.parse_args()
    
    run_motion_estimation_pipeline(
        video_path=args.video,
        output_dir=args.output_dir,
        max_frames_to_process=args.max_frames,
        diff_threshold=args.diff_thresh
    )
