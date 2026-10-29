"""
Task 2: Feature Extraction and Image Matching.

This module implements:
1. Histogram of Oriented Gradients (HOG) feature computation and visualization.
2. Scale-Invariant Feature Transform (SIFT) keypoint detection and 128-D descriptor extraction.
3. SIFT keypoint visualization (rich keypoints with scale and orientation).
4. Feature matching using Brute-Force Matcher with Lowe's ratio test filtering.
5. Scale and rotation invariance experiment (rotating and resizing the second photo).
6. Robust fallback handling (ORB) if SIFT is unavailable, and graceful warnings
   if images have low texture or insufficient keypoints.
7. Quantitative reporting of keypoints, raw matches, and good matches.
"""

import sys
import os

# Ensure project root is in sys.path when running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

from src.utils import (
    load_rgb,
    load_gray,
    save_figure,
    save_json,
    save_csv_table,
    print_banner,
    print_step,
    ensure_dir,
)
from src.validators import validate_photo_original, validate_photo_second, InputValidationError


def compute_and_visualize_hog(
    image1_rgb: np.ndarray,
    image2_rgb: np.ndarray,
    photo1_name: str = "Image 1",
    photo2_name: str = "Image 2",
    output_dir: str = "outputs/task2"
) -> str:
    """
    Computes Histogram of Oriented Gradients (HOG) features for both images
    and generates side-by-side gradient orientation visualizations.
    """
    print_step(1, "Computing Histogram of Oriented Gradients (HOG) Features")
    ensure_dir(output_dir)
    
    # Convert to grayscale for standard HOG computation
    gray1 = cv2.cvtColor(image1_rgb, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2_rgb, cv2.COLOR_RGB2GRAY)
    
    # Resize to moderate resolution for responsive and clear HOG visualization
    target_size = (512, 512)
    gray1_resized = cv2.resize(gray1, target_size, interpolation=cv2.INTER_AREA)
    gray2_resized = cv2.resize(gray2, target_size, interpolation=cv2.INTER_AREA)
    
    # Compute HOG descriptors and visualization images
    # 9 orientation bins, 16x16 pixel cells, 2x2 cells per block
    fd1, hog_img1 = hog(
        gray1_resized,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm='L2-Hys'
    )
    hog_img1_rescaled = exposure.rescale_intensity(hog_img1, in_range=(0, 10))
    
    fd2, hog_img2 = hog(
        gray2_resized,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm='L2-Hys'
    )
    hog_img2_rescaled = exposure.rescale_intensity(hog_img2, in_range=(0, 10))
    
    print(f"   -> Image 1 ({photo1_name}) HOG Feature Vector Length: {len(fd1)} dimensions")
    print(f"   -> Image 2 ({photo2_name}) HOG Feature Vector Length: {len(fd2)} dimensions")
    
    # Plot side-by-side
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Task 2: Histogram of Oriented Gradients (HOG) Feature Representation", fontsize=15, fontweight="bold")
    
    axes[0, 0].imshow(cv2.resize(image1_rgb, target_size))
    axes[0, 0].set_title(f"1. {photo1_name} (Resized 512x512)")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(hog_img1_rescaled, cmap="inferno")
    axes[0, 1].set_title(f"2. {photo1_name} HOG Gradient Map\n({len(fd1)} Feature Dimensions)")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(cv2.resize(image2_rgb, target_size))
    axes[1, 0].set_title(f"3. {photo2_name} (Resized 512x512)")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(hog_img2_rescaled, cmap="inferno")
    axes[1, 1].set_title(f"4. {photo2_name} HOG Gradient Map\n({len(fd2)} Feature Dimensions)")
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "hog_feature_visualization.png")
    save_figure(fig, fig_path)
    return fig_path


def get_feature_detector() -> Tuple[Any, str, int]:
    """
    Initializes SIFT detector, falling back gracefully to ORB if SIFT
    is unavailable in the current OpenCV installation.
    
    Returns:
        (detector_instance, algorithm_name, norm_type)
    """
    try:
        sift = cv2.SIFT_create()
        return sift, "SIFT", cv2.NORM_L2
    except Exception as e:
        print(f"\n   [WARNING]: cv2.SIFT_create() unavailable ({e}). Falling back to ORB detector.")
        orb = cv2.ORB_create(nfeatures=2000)
        return orb, "ORB", cv2.NORM_HAMMING


def save_sift_keypoints_visualization(
    image1_rgb: np.ndarray,
    image2_rgb: np.ndarray,
    kp1: List[Any],
    kp2: List[Any],
    algo_name: str,
    photo1_name: str = "Image 1",
    photo2_name: str = "Image 2",
    output_dir: str = "outputs/task2"
) -> str:
    """
    Draws and saves rich keypoint visualizations displaying keypoint locations,
    scale (circle size), and dominant orientation angle.
    """
    img1_bgr = cv2.cvtColor(image1_rgb, cv2.COLOR_RGB2BGR)
    img2_bgr = cv2.cvtColor(image2_rgb, cv2.COLOR_RGB2BGR)
    
    # Draw rich keypoints with scale and orientation
    vis1_bgr = cv2.drawKeypoints(
        img1_bgr, kp1, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)
    )
    vis2_bgr = cv2.drawKeypoints(
        img2_bgr, kp2, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)
    )
    
    vis1_rgb = cv2.cvtColor(vis1_bgr, cv2.COLOR_BGR2RGB)
    vis2_rgb = cv2.cvtColor(vis2_bgr, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Task 2: {algo_name} Keypoint Detection (Rich Keypoints with Scale & Orientation)", fontsize=14, fontweight="bold")
    
    axes[0].imshow(vis1_rgb)
    axes[0].set_title(f"1. {photo1_name} ({len(kp1)} keypoints detected)", fontsize=12)
    axes[0].axis("off")
    
    axes[1].imshow(vis2_rgb)
    axes[1].set_title(f"2. {photo2_name} ({len(kp2)} keypoints detected)", fontsize=12)
    axes[1].axis("off")
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "sift_keypoints.png")
    save_figure(fig, fig_path)
    return fig_path


def match_keypoints(
    detector: Any,
    norm_type: int,
    gray1: np.ndarray,
    gray2: np.ndarray,
    ratio_threshold: float = 0.75
) -> Tuple[List[Any], List[Any], Optional[np.ndarray], Optional[np.ndarray], List[Any], float]:
    """
    Detects keypoints, computes descriptors, and matches them using BFMatcher
    with Lowe's ratio test filtering.
    """
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    
    # Check keypoint counts
    if len(kp1) < 10:
        print(f"   [WARNING]: Low keypoint count in Image 1 ({len(kp1)} keypoints). Check lighting and texture.")
    if len(kp2) < 10:
        print(f"   [WARNING]: Low keypoint count in Image 2 ({len(kp2)} keypoints). Check lighting and texture.")
        
    # Handle cases where descriptors cannot be computed
    if des1 is None or len(kp1) == 0:
        print("   [ERROR]: No feature descriptors found in Image 1.")
        return kp1, kp2, des1, des2, [], 0.0
    if des2 is None or len(kp2) == 0:
        print("   [ERROR]: No feature descriptors found in Image 2.")
        return kp1, kp2, des1, des2, [], 0.0
        
    bf = cv2.BFMatcher(norm_type)
    # k=2 for Lowe's ratio test
    raw_matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test: m.distance < ratio_threshold * n.distance
    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
                
    if len(good_matches) == 0 and len(raw_matches) > 0:
        print(f"   [WARNING]: 0 matches passed Lowe's ratio test (threshold={ratio_threshold}). All {len(raw_matches)} candidates filtered out.")
        
    min_kps = min(len(kp1), len(kp2))
    match_ratio = (len(good_matches) / min_kps * 100.0) if min_kps > 0 else 0.0
    
    return kp1, kp2, des1, des2, good_matches, match_ratio


def apply_geometric_transform(
    image: np.ndarray,
    angle_degrees: float = 45.0,
    scale_factor: float = 0.7
) -> np.ndarray:
    """
    Applies synthetic 2D rotation and uniform scaling to test transformation invariance.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 2D Affine Rotation matrix with scaling
    M = cv2.getRotationMatrix2D(center, angle_degrees, scale_factor)
    transformed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return transformed


def run_feature_matching_pipeline(
    photo1_path: str = "inputs/photo_original.jpg",
    photo2_path: str = "inputs/photo_second.jpg",
    output_dir: str = "outputs/task2",
    ratio_threshold: float = 0.75,
    test_rotation_deg: float = 45.0,
    test_scale_factor: float = 0.7
) -> Dict[str, Any]:
    """
    Executes the complete Task 2 pipeline:
    - Input validation and image loading with clear errors
    - Output directory creation upfront
    - HOG extraction and visualization
    - SIFT keypoint detection, description, and keypoint visualization
    - Lowe's ratio matching on baseline photo pair
    - Geometric transformation (rotation & scale) on photo 2
    - SIFT matching on transformed photo pair to verify scale/rotation invariance
    - Metric logging and comparative reporting
    """
    print_banner(
        "TASK 2: FEATURE EXTRACTION AND IMAGE MATCHING",
        f"Pair: {photo1_path} & {photo2_path} | Output Dir: {output_dir}"
    )
    
    # Ensure output directory exists before any saving
    output_dir = ensure_dir(output_dir)
    print(f" Verified output directory: '{output_dir}'")
    
    # 1. Validate inputs exist on disk
    if not os.path.exists(photo1_path):
        msg = (
            f"\n[MISSING INPUT ERROR]: Image 1 not found at: '{photo1_path}'.\n"
            f"Please ensure your primary photograph exists. See inputs/README.md for details."
        )
        raise InputValidationError(msg)
        
    if not os.path.exists(photo2_path):
        msg = (
            f"\n[MISSING INPUT ERROR]: Image 2 not found at: '{photo2_path}'.\n"
            f"Task 2 requires two photos of the same subject or scene for feature matching.\n"
            f"To fix this:\n"
            f"  1. Add your second photo as '{photo2_path}'.\n"
            f"  2. Or specify an existing photo using --photo2 <path> / --img2 <path>.\n"
            f"See inputs/README.md for guidelines on capturing a matching photo pair."
        )
        raise InputValidationError(msg)
        
    # Load images
    try:
        img1_rgb = load_rgb(photo1_path)
    except Exception as e:
        raise ValueError(f"[IMAGE LOAD ERROR] Failed to decode Image 1 '{photo1_path}': {e}")
        
    try:
        img2_rgb = load_rgb(photo2_path)
    except Exception as e:
        raise ValueError(f"[IMAGE LOAD ERROR] Failed to decode Image 2 '{photo2_path}': {e}")
        
    name1 = os.path.basename(photo1_path)
    name2 = os.path.basename(photo2_path)
    print(f" Loaded '{name1}': {img1_rgb.shape[1]}x{img1_rgb.shape[0]} px")
    print(f" Loaded '{name2}': {img2_rgb.shape[1]}x{img2_rgb.shape[0]} px")
    
    gray1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
    
    # 2. HOG Feature Visualization
    hog_fig = compute_and_visualize_hog(img1_rgb, img2_rgb, photo1_name=name1, photo2_name=name2, output_dir=output_dir)
    
    # 3. Initialize Feature Detector (SIFT or ORB fallback)
    detector, algo_name, norm_type = get_feature_detector()
    print_step(2, f"Detecting Keypoints and Extracting Descriptors using {algo_name}")
    print(f"   -> Algorithm: {algo_name}")
    print(f"   -> Matcher: Brute-Force Matcher with Lowe's Ratio Test (Threshold = {ratio_threshold})")
    
    # Test A: Baseline Pair (Photo 1 vs Photo 2)
    print(f"\n   [EXPERIMENT A: Real-World Photo Pair ({name1} vs {name2})]")
    kp1, kp2, des1, des2, good_matches_base, match_ratio_base = match_keypoints(
        detector, norm_type, gray1, gray2, ratio_threshold=ratio_threshold
    )
    
    print(f"      - Image 1 ({name1}) Keypoints: {len(kp1)}")
    print(f"      - Image 2 ({name2}) Keypoints: {len(kp2)}")
    print(f"      - Good Matches (passed ratio test): {len(good_matches_base)}")
    print(f"      - Good Match Ratio: {match_ratio_base:.2f}% of min keypoints")
    
    # Save SIFT Keypoints Visualization
    sift_kp_path = save_sift_keypoints_visualization(
        img1_rgb, img2_rgb, kp1, kp2, algo_name=algo_name,
        photo1_name=name1, photo2_name=name2, output_dir=output_dir
    )
    
    # Draw Baseline Matches Visualization (show top 60 matches for visual clarity)
    img1_bgr = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
    img2_bgr = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR)
    
    sorted_matches_base = sorted(good_matches_base, key=lambda m: m.distance)
    matches_to_draw_base = sorted_matches_base[:min(60, len(sorted_matches_base))]
    
    vis_base_bgr = cv2.drawMatches(
        img1_bgr, kp1,
        img2_bgr, kp2,
        matches_to_draw_base, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0)
    )
    vis_base_rgb = cv2.cvtColor(vis_base_bgr, cv2.COLOR_BGR2RGB)
    
    fig_base, ax_base = plt.subplots(figsize=(18, 8))
    ax_base.imshow(vis_base_rgb)
    ax_base.set_title(
        f"Task 2: {algo_name} Feature Matching — Baseline Pair ({name1} vs {name2})\n"
        f"Keypoints: (Img1={len(kp1)}, Img2={len(kp2)}) | Good Matches: {len(good_matches_base)} "
        f"({match_ratio_base:.1f}% Match Ratio)",
        fontsize=13, fontweight="bold"
    )
    ax_base.axis("off")
    plt.tight_layout()
    fig_base_path = os.path.join(output_dir, "sift_matches_baseline.png")
    save_figure(fig_base, fig_base_path)
    
    # Test B: Rotation & Scale Invariance Test
    print_step(3, f"Testing {algo_name} Invariance Under Rotation ({test_rotation_deg}°) and Scale ({test_scale_factor}x)")
    img2_transformed = apply_geometric_transform(img2_rgb, angle_degrees=test_rotation_deg, scale_factor=test_scale_factor)
    gray2_transformed = cv2.cvtColor(img2_transformed, cv2.COLOR_RGB2GRAY)
    
    kp1_t, kp2_t, des1_t, des2_t, good_matches_trans, match_ratio_trans = match_keypoints(
        detector, norm_type, gray1, gray2_transformed, ratio_threshold=ratio_threshold
    )
    
    print(f"      - Image 1 Keypoints: {len(kp1_t)}")
    print(f"      - Transformed Image 2 Keypoints: {len(kp2_t)}")
    print(f"      - Good Matches (passed ratio test): {len(good_matches_trans)}")
    print(f"      - Good Match Ratio: {match_ratio_trans:.2f}% of min keypoints")
    
    # Draw Transformed Matches Visualization
    img2_trans_bgr = cv2.cvtColor(img2_transformed, cv2.COLOR_RGB2BGR)
    sorted_matches_trans = sorted(good_matches_trans, key=lambda m: m.distance)
    matches_to_draw_trans = sorted_matches_trans[:min(60, len(sorted_matches_trans))]
    
    vis_trans_bgr = cv2.drawMatches(
        img1_bgr, kp1_t,
        img2_trans_bgr, kp2_t,
        matches_to_draw_trans, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(255, 128, 0)
    )
    vis_trans_rgb = cv2.cvtColor(vis_trans_bgr, cv2.COLOR_BGR2RGB)
    
    fig_trans, ax_trans = plt.subplots(figsize=(18, 8))
    ax_trans.imshow(vis_trans_rgb)
    ax_trans.set_title(
        f"Task 2: {algo_name} Invariance Test — {name1} vs Transformed {name2} (Rotated {test_rotation_deg}°, Scale {test_scale_factor}x)\n"
        f"Keypoints: (Img1={len(kp1_t)}, Transformed Img2={len(kp2_t)}) | Good Matches: {len(good_matches_trans)} "
        f"({match_ratio_trans:.1f}% Match Ratio)",
        fontsize=13, fontweight="bold"
    )
    ax_trans.axis("off")
    plt.tight_layout()
    fig_trans_path = os.path.join(output_dir, "sift_matches_transformed.png")
    save_figure(fig_trans, fig_trans_path)
    
    # Summary Table Records
    records = [
        {
            "Test_Condition": f"Baseline ({name1} vs {name2})",
            "Detector": algo_name,
            "Img1_Keypoints": len(kp1),
            "Img2_Keypoints": len(kp2),
            "Good_Matches": len(good_matches_base),
            "Good_Match_Ratio_Pct": round(match_ratio_base, 2),
            "Transformation": "Real-world viewpoint/scale variation"
        },
        {
            "Test_Condition": f"Synthetic Transform (Rotated {test_rotation_deg}°, Scale {test_scale_factor}x)",
            "Detector": algo_name,
            "Img1_Keypoints": len(kp1_t),
            "Img2_Keypoints": len(kp2_t),
            "Good_Matches": len(good_matches_trans),
            "Good_Match_Ratio_Pct": round(match_ratio_trans, 2),
            "Transformation": f"Rotation={test_rotation_deg} deg, Scale={test_scale_factor}"
        }
    ]
    
    csv_path = os.path.join(output_dir, "matching_metrics.csv")
    json_path = os.path.join(output_dir, "matching_metrics.json")
    save_csv_table(records, csv_path)
    save_json(records, json_path)
    
    # Bar Chart Comparison
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    conditions = ["Baseline Pair", f"Transformed ({test_rotation_deg}°, {test_scale_factor}x)"]
    good_counts = [len(good_matches_base), len(good_matches_trans)]
    colors = ["#2ca02c", "#ff7f0e"]
    
    bars = ax_bar.bar(conditions, good_counts, color=colors, width=0.45)
    ax_bar.set_ylabel("Good Matches Count (Lowe's Ratio < 0.75)")
    ax_bar.set_title(f"Task 2: Quantitative Feature Matching Comparison ({algo_name})", fontsize=14, fontweight="bold")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2.0, yval + max(1, yval * 0.02), f"{int(yval)}", ha="center", va="bottom", fontweight="bold")
        
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "matching_comparison_summary.png")
    save_figure(fig_bar, bar_path)
    
    print("\n" + "=" * 76)
    print(" [COMPLETED] Task 2 pipeline finished successfully!")
    print(" Generated Output Artifacts:")
    print(f"   1. HOG Feature Visualization : {hog_fig}")
    print(f"   2. SIFT Keypoints Image      : {sift_kp_path}")
    print(f"   3. Baseline Feature Matches  : {fig_base_path}")
    print(f"   4. Transformed SIFT Matches  : {fig_trans_path}")
    print(f"   5. Quantitative Summary Plot : {bar_path}")
    print(f"   6. Metrics Table (CSV)       : {csv_path}")
    print(f"   7. Metrics Data (JSON)       : {json_path}")
    print("=" * 76 + "\n")
    
    return {
        "algorithm": algo_name,
        "baseline_matches": len(good_matches_base),
        "transformed_matches": len(good_matches_trans),
        "metrics_csv": csv_path,
        "figures": [hog_fig, sift_kp_path, fig_base_path, fig_trans_path, bar_path]
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Task 2: Feature Extraction and Image Matching")
    parser.add_argument("--img1", default="inputs/photo_original.jpg", help="Path to primary photograph")
    parser.add_argument("--img2", default="inputs/photo_second.jpg", help="Path to second photograph of same subject")
    parser.add_argument("--output-dir", default="outputs/task2", help="Directory to save task outputs")
    parser.add_argument("--ratio-thresh", type=float, default=0.75, help="Lowe's ratio test threshold")
    parser.add_argument("--rotation", type=float, default=45.0, help="Rotation angle in degrees for invariance test")
    parser.add_argument("--scale", type=float, default=0.7, help="Scale factor for invariance test")
    args = parser.parse_args()
    
    try:
        run_feature_matching_pipeline(
            photo1_path=args.img1,
            photo2_path=args.img2,
            output_dir=args.output_dir,
            ratio_threshold=args.ratio_thresh,
            test_rotation_deg=args.rotation,
            test_scale_factor=args.scale
        )
    except (InputValidationError, FileNotFoundError, ValueError) as e:
        print(f"\n{e}\n")
        sys.exit(1)
