"""
Task 1: Image Restoration and Segmentation.

This module implements:
1. Reproducible Gaussian and Salt-and-Pepper noise addition.
2. Image restoration using Gaussian and Median spatial filters.
3. Quantitative comparison using PSNR (dB) and SSIM metrics.
4. Binary thresholding (Otsu's method) and morphological operations
   (erosion, dilation, opening, closing).
5. Multi-scale Canny edge detection.
6. Marker-controlled Watershed segmentation with suitability diagnostics
   and adjustable parameters.
"""

import os
import sys

# Ensure project root is in sys.path when running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

from src.utils import (
    load_rgb,
    load_gray,
    save_figure,
    save_json,
    save_csv_table,
    print_banner,
    print_step,
    print_metric_explanation,
    ensure_dir,
)
from src.validators import validate_photo_original


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 25.0, seed: int = 42) -> np.ndarray:
    """
    Adds zero-mean Gaussian noise to an image with fixed seed for reproducibility.
    
    Mathematical Model:
        I_noisy(x, y) = clip(I_clean(x, y) + N(mean, sigma^2), 0, 255)
    """
    rng = np.random.RandomState(seed)
    noise = rng.normal(mean, sigma, image.shape)
    noisy_img = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy_img


def add_salt_and_pepper_noise(
    image: np.ndarray,
    salt_prob: float = 0.02,
    pepper_prob: float = 0.02,
    seed: int = 42
) -> np.ndarray:
    """
    Adds impulse (Salt-and-Pepper) noise with fixed seed for reproducibility.
    
    Mathematical Model:
        Pixel becomes 255 with probability P_salt (Salt / white saturated pixel),
        0 with probability P_pepper (Pepper / dark saturated pixel),
        or remains unchanged with probability (1 - P_salt - P_pepper).
    """
    rng = np.random.RandomState(seed)
    noisy_img = image.copy()
    
    # Generate random uniform matrix for coordinates
    random_matrix = rng.rand(image.shape[0], image.shape[1])
    
    # Salt noise (white pixels)
    salt_mask = random_matrix < salt_prob
    noisy_img[salt_mask] = 255
    
    # Pepper noise (black pixels)
    pepper_mask = (random_matrix >= salt_prob) & (random_matrix < (salt_prob + pepper_prob))
    noisy_img[pepper_mask] = 0
    
    return noisy_img


def calculate_metrics(clean: np.ndarray, degraded_or_restored: np.ndarray) -> Tuple[float, float]:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR in dB) and Structural Similarity (SSIM).
    
    Both inputs must have the same shape and uint8 dtype.
    """
    # PSNR calculation
    psnr_val = float(psnr_func(clean, degraded_or_restored, data_range=255))
    
    # SSIM calculation (multichannel if color image)
    if clean.ndim == 3:
        ssim_val = float(ssim_func(clean, degraded_or_restored, data_range=255, channel_axis=2))
    else:
        ssim_val = float(ssim_func(clean, degraded_or_restored, data_range=255))
        
    return psnr_val, ssim_val


def run_noise_and_restoration(
    image_rgb: np.ndarray,
    output_dir: str = "outputs/task1"
) -> Dict[str, Any]:
    """
    Applies Gaussian and S&P noise, filters them with Gaussian and Median filters,
    and calculates comparative PSNR and SSIM metrics.
    """
    print_step(1, "Generating Reproducible Noise and Applying Restoration Filters")
    ensure_dir(output_dir)
    
    # 1. Add noise
    print("   -> Injecting Gaussian noise (mean=0, sigma=25, seed=42)...")
    gaussian_noisy = add_gaussian_noise(image_rgb, mean=0.0, sigma=25.0, seed=42)
    
    print("   -> Injecting Salt-and-Pepper noise (salt=2%, pepper=2%, seed=42)...")
    sp_noisy = add_salt_and_pepper_noise(image_rgb, salt_prob=0.02, pepper_prob=0.02, seed=42)
    
    # 2. Apply filters
    print("   -> Applying Gaussian Filter (5x5 kernel, sigma=1.2)...")
    gauss_on_gauss = cv2.GaussianBlur(gaussian_noisy, (5, 5), 1.2)
    gauss_on_sp = cv2.GaussianBlur(sp_noisy, (5, 5), 1.2)
    
    print("   -> Applying Median Filter (5x5 neighborhood)...")
    median_on_gauss = cv2.medianBlur(gaussian_noisy, 5)
    median_on_sp = cv2.medianBlur(sp_noisy, 5)
    
    # 3. Compute Metrics
    print("   -> Calculating PSNR (dB) and SSIM metrics...")
    experiments = [
        ("Gaussian Noise (Raw)", gaussian_noisy, "Baseline Gaussian corruption"),
        ("Gaussian Filter on Gaussian Noise", gauss_on_gauss, "Linear smoothing of Gaussian noise"),
        ("Median Filter on Gaussian Noise", median_on_gauss, "Non-linear order-statistic filtering"),
        ("Salt-and-Pepper Noise (Raw)", sp_noisy, "Baseline impulse corruption"),
        ("Gaussian Filter on S&P Noise", gauss_on_sp, "Linear smoothing of impulse spikes (blurs salt/pepper)"),
        ("Median Filter on S&P Noise", median_on_sp, "Rank-order median replaces extreme outliers perfectly"),
    ]
    
    records = []
    for name, img, notes in experiments:
        psnr_val, ssim_val = calculate_metrics(image_rgb, img)
        records.append({
            "Experiment": name,
            "PSNR_dB": round(psnr_val, 2),
            "SSIM": round(ssim_val, 4),
            "Notes": notes
        })
        print(f"      [{name:<34}] -> PSNR: {psnr_val:6.2f} dB | SSIM: {ssim_val:.4f}")

    # Save metrics table
    csv_path = os.path.join(output_dir, "restoration_metrics.csv")
    json_path = os.path.join(output_dir, "restoration_metrics.json")
    save_csv_table(records, csv_path)
    save_json(records, json_path)
    
    # 4. Generate 2x4 comparison figure
    print("   -> Creating multi-panel comparison figure...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Task 1: Image Noise Addition & Restoration Filter Comparison", fontsize=16, fontweight="bold", y=0.98)
    
    # Row 1: Gaussian Noise & Restorations
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("1. Original Clean Photo\n(Reference Ground Truth)")
    axes[0, 0].axis("off")
    
    p, s = records[0]["PSNR_dB"], records[0]["SSIM"]
    axes[0, 1].imshow(gaussian_noisy)
    axes[0, 1].set_title(f"2. Gaussian Noise (sigma=25)\nPSNR: {p:.2f} dB | SSIM: {s:.4f}")
    axes[0, 1].axis("off")
    
    p, s = records[1]["PSNR_dB"], records[1]["SSIM"]
    axes[0, 2].imshow(gauss_on_gauss)
    axes[0, 2].set_title(f"3. Gaussian Filter on Gauss Noise\nPSNR: {p:.2f} dB | SSIM: {s:.4f}")
    axes[0, 2].axis("off")
    
    p, s = records[2]["PSNR_dB"], records[2]["SSIM"]
    axes[0, 3].imshow(median_on_gauss)
    axes[0, 3].set_title(f"4. Median Filter on Gauss Noise\nPSNR: {p:.2f} dB | SSIM: {s:.4f}")
    axes[0, 3].axis("off")
    
    # Row 2: Salt & Pepper Noise & Restorations
    axes[1, 0].imshow(image_rgb)
    axes[1, 0].set_title("5. Original Clean Photo\n(Reference Ground Truth)")
    axes[1, 0].axis("off")
    
    p, s = records[3]["PSNR_dB"], records[3]["SSIM"]
    axes[1, 1].imshow(sp_noisy)
    axes[1, 1].set_title(f"6. Salt & Pepper Noise (4% total)\nPSNR: {p:.2f} dB | SSIM: {s:.4f}")
    axes[1, 1].axis("off")
    
    p, s = records[4]["PSNR_dB"], records[4]["SSIM"]
    axes[1, 2].imshow(gauss_on_sp)
    axes[1, 2].set_title(f"7. Gaussian Filter on S&P Noise\nPSNR: {p:.2f} dB | SSIM: {s:.4f}\n(Blurs impulses into smudges)")
    axes[1, 2].axis("off")
    
    p, s = records[5]["PSNR_dB"], records[5]["SSIM"]
    axes[1, 3].imshow(median_on_sp)
    axes[1, 3].set_title(f"8. Median Filter on S&P Noise\nPSNR: {p:.2f} dB | SSIM: {s:.4f}\n(Optimal: removes impulse outliers)")
    axes[1, 3].axis("off")
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "noise_and_restoration_comparison.png")
    save_figure(fig, fig_path)
    
    print_metric_explanation()
    return {"metrics": records, "figure": fig_path}


def run_thresholding_and_morphology(
    image_gray: np.ndarray,
    output_dir: str = "outputs/task1"
) -> Dict[str, Any]:
    """
    Applies Otsu's automatic binary thresholding and demonstrates the four
    fundamental morphological operations: Erosion, Dilation, Opening, Closing.
    """
    print_step(2, "Applying Binary Thresholding and Morphological Operations")
    ensure_dir(output_dir)
    
    # 1. Otsu's thresholding
    # Minimizes intra-class variance between foreground and background pixels
    otsu_thresh_val, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"   -> Calculated optimal Otsu threshold value: {otsu_thresh_val:.1f}")
    
    # 2. Structuring Element (Kernel)
    # 5x5 elliptical structuring element for natural, isotropic morphological transformations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # 3. Morphological Operations
    # Erosion: Computes local minimum; shrinks bright regions, strips small isolated noise protrusions
    erosion = cv2.erode(binary, kernel, iterations=1)
    
    # Dilation: Computes local maximum; expands bright regions, fills small dark voids and holes
    dilation = cv2.dilate(binary, kernel, iterations=1)
    
    # Opening: Erosion followed by Dilation; eliminates small foreground noise/islands while preserving overall size
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Closing: Dilation followed by Erosion; fills small interior holes and bridges narrow gaps between adjacent components
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 4. Visualization plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Task 1: Binary Thresholding & Morphological Operations (Otsu Thresh = {otsu_thresh_val:.1f})",
                 fontsize=15, fontweight="bold")
    
    axes[0, 0].imshow(image_gray, cmap="gray")
    axes[0, 0].set_title("1. Original Grayscale Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(binary, cmap="gray")
    axes[0, 1].set_title("2. Otsu's Binary Thresholded Image")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(erosion, cmap="gray")
    axes[0, 2].set_title("3. Erosion (Shrinks bright objects,\neliminates thin connections)")
    axes[0, 2].axis("off")
    
    axes[1, 0].imshow(dilation, cmap="gray")
    axes[1, 0].set_title("4. Dilation (Expands bright objects,\nfills small internal holes)")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(opening, cmap="gray")
    axes[1, 1].set_title("5. Opening (Erode -> Dilate:\nRemoves small isolated noise)")
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(closing, cmap="gray")
    axes[1, 2].set_title("6. Closing (Dilate -> Erode:\nBridges narrow gaps and cracks)")
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "morphology_operations.png")
    save_figure(fig, fig_path)
    
    return {
        "otsu_threshold": otsu_thresh_val,
        "figure": fig_path
    }


def run_canny_edge_detection(
    image_gray: np.ndarray,
    output_dir: str = "outputs/task1"
) -> Dict[str, Any]:
    """
    Applies Canny edge detection across different hysteresis thresholds to illustrate
    the trade-off between edge connectivity and noise sensitivity.
    """
    print_step(3, "Applying Canny Multi-Stage Edge Detection")
    ensure_dir(output_dir)
    
    # Pre-filtering with Gaussian blur to suppress noise gradients
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 1.4)
    
    # Calculate median-based automatic thresholding (Otsu-style heuristic)
    med = np.median(blurred)
    auto_low = int(max(0, (1.0 - 0.33) * med))
    auto_high = int(min(255, (1.0 + 0.33) * med))
    
    # Edge maps with distinct parameter settings
    # 1. Conservative (strict thresholds - only strong edges detected)
    canny_strict = cv2.Canny(blurred, 100, 200)
    # 2. Balanced (standard thresholds)
    canny_balanced = cv2.Canny(blurred, 50, 150)
    # 3. Sensitive (low thresholds - detects fine details and texture edges)
    canny_sensitive = cv2.Canny(blurred, 25, 75)
    # 4. Automatic heuristic
    canny_auto = cv2.Canny(blurred, auto_low, auto_high)
    
    print(f"   -> Canny thresholds computed:")
    print(f"      - Strict:   Low=100, High=200")
    print(f"      - Balanced: Low=50,  High=150")
    print(f"      - Sensitive:Low=25,  High=75")
    print(f"      - Auto:     Low={auto_low}, High={auto_high} (based on median intensity {med:.1f})")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Task 1: Canny Edge Detection across Hysteresis Thresholds", fontsize=15, fontweight="bold")
    
    axes[0].imshow(canny_strict, cmap="gray")
    axes[0].set_title("1. Strict Thresholds (100, 200)\n(Only most prominent contours)")
    axes[0].axis("off")
    
    axes[1].imshow(canny_balanced, cmap="gray")
    axes[1].set_title("2. Balanced Standard (50, 150)\n(Clean object boundaries)")
    axes[1].axis("off")
    
    axes[2].imshow(canny_sensitive, cmap="gray")
    axes[2].set_title("3. Sensitive (25, 75)\n(Picks up subtle textures)")
    axes[2].axis("off")
    
    axes[3].imshow(canny_auto, cmap="gray")
    axes[3].set_title(f"4. Auto-Tuned ({auto_low}, {auto_high})\n(Adaptive to median intensity)")
    axes[3].axis("off")
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "canny_edge_detection.png")
    save_figure(fig, fig_path)
    
    return {"figure": fig_path}


def run_watershed_segmentation(
    image_rgb: np.ndarray,
    output_dir: str = "outputs/task1",
    dist_threshold_ratio: float = 0.5,
    blur_ksize: int = 5,
    morph_iter: int = 2
) -> Dict[str, Any]:
    """
    Applies Marker-Controlled Watershed Segmentation to segment touching objects.
    
    Workflow:
      1. Preprocessing (Grayscale + Gaussian Blur).
      2. Otsu thresholding -> Binary mask.
      3. Morphological opening to remove spurious foreground dots.
      4. Dilation to identify sure background.
      5. Euclidean Distance Transform (cv2.distanceTransform) to find object centers.
      6. Threshold distance transform to obtain sure foreground markers.
      7. Unknown region = sure_bg - sure_fg.
      8. Connected components labeling to assign unique IDs to each object peak.
      9. Run cv2.watershed on original color image.
      10. Mark watershed boundary lines in bright red.
      
    Suitability Diagnostics:
      If the image has too few or too many markers, outputs a comprehensive diagnosis
      explaining what makes the photo unsuitable and which settings to adjust.
    """
    print_step(4, "Performing Marker-Controlled Watershed Segmentation")
    ensure_dir(output_dir)
    
    # Step 1: Grayscale and blur
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    
    # Step 2: Otsu binarization
    # Invert if the foreground objects are darker than background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Check if inversion was appropriate: if more than 60% of pixels are white, invert back
    if np.mean(thresh == 255) > 0.60:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    # Step 3: Morphological opening to eliminate isolated pixel specks
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
    
    # Step 4: Sure background area via dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Step 5: Finding sure foreground area using Euclidean Distance Transform
    # Measures Euclidean distance of each foreground pixel to nearest background pixel
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_dist = dist_transform.max()
    
    # Step 6: Sure foreground markers
    if max_dist > 0:
        _, sure_fg = cv2.threshold(dist_transform, dist_threshold_ratio * max_dist, 255, 0)
    else:
        sure_fg = np.zeros_like(opening)
    sure_fg = np.uint8(sure_fg)
    
    # Step 7: Unknown boundary region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Step 8: Marker labeling
    # cv2.connectedComponents assigns unique positive integer labels (1, 2, ..., N) to each peak
    num_markers, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Mark the unknown region as 0 for watershed to flood
    markers[unknown == 255] = 0
    
    # Step 9: Run Watershed algorithm
    # Note: cv2.watershed requires an 8-bit 3-channel BGR image
    bgr_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    markers_result = cv2.watershed(bgr_img, markers)
    
    # Step 10: Overlay watershed boundary lines on original image
    watershed_overlay = image_rgb.copy()
    # -1 represents the watershed boundary ridge lines
    watershed_overlay[markers_result == -1] = [255, 0, 0]  # Bright red boundaries
    
    # Diagnostic assessment
    distinct_objects = num_markers - 1  # Exclude background component
    print(f"   -> Distance Transform Peak Max Value: {max_dist:.2f}")
    print(f"   -> Distinct Foreground Object Markers Detected: {distinct_objects}")
    
    suitability_notes = []
    if distinct_objects < 2:
        msg = (
            f"\n   [WATERSHED SUITABILITY DIAGNOSTIC: FEW / NO DISTINCT PEAKS DETECTED]\n"
            f"   Detected only {distinct_objects} foreground marker(s).\n"
            f"   Why this image may be challenging for Watershed:\n"
            f"     1. Low contrast between objects and background prevented Otsu's\n"
            f"        threshold from segmenting foreground cleanly.\n"
            f"     2. Flat lighting or non-uniform gradient across the photograph.\n"
            f"     3. The photograph might contain a single continuous scene rather\n"
            f"        than multiple distinct objects.\n"
            f"   How to adjust or improve:\n"
            f"     - Try adjusting '--watershed-dist-ratio' (e.g. lower to 0.3 for weaker peaks).\n"
            f"     - For best watershed results, photograph multiple distinct items\n"
            f"       (such as coins, keys, buttons, or fruit) on a high-contrast plain surface.\n"
        )
        print(msg)
        suitability_notes.append(msg)
    elif distinct_objects > 100:
        msg = (
            f"\n   [WATERSHED SUITABILITY DIAGNOSTIC: OVER-SEGMENTATION OBSERVED]\n"
            f"   Detected {distinct_objects} markers (extreme over-segmentation).\n"
            f"   Why this happens:\n"
            f"     High-frequency textures (e.g. hair, fabric, foliage, carpet)\n"
            f"     produce hundreds of local distance transform maxima.\n"
            f"   How to adjust or improve:\n"
            f"     - Increase '--watershed-dist-ratio' (e.g. 0.6 or 0.7).\n"
            f"     - Increase '--watershed-blur' (e.g. 7 or 9) to smooth local variations.\n"
        )
        print(msg)
        suitability_notes.append(msg)
    else:
        msg = f"   [SUCCESS] Suitable marker topology detected ({distinct_objects} separated object regions)."
        print(msg)
        suitability_notes.append(msg)

    # Visualization plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Task 1: Marker-Controlled Watershed Segmentation", fontsize=15, fontweight="bold")
    
    axes[0, 0].imshow(thresh, cmap="gray")
    axes[0, 0].set_title("1. Binarized Image (Otsu + Morphology)")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(dist_transform, cmap="viridis")
    axes[0, 1].set_title(f"2. Euclidean Distance Transform\n(Peak Value: {max_dist:.1f})")
    axes[0, 1].axis("off")
    
    # Colored markers map
    axes[1, 0].imshow(markers_result, cmap="nipy_spectral")
    axes[1, 0].set_title(f"3. Watershed Catchment Basins\n({distinct_objects} Object Markers)")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(watershed_overlay)
    axes[1, 1].set_title("4. Final Watershed Boundaries (Red Ridges)\nOverlaid on Original Image")
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "watershed_segmentation.png")
    save_figure(fig, fig_path)
    
    summary = {
        "distinct_object_markers": distinct_objects,
        "max_distance_transform": float(round(max_dist, 2)),
        "dist_threshold_ratio_used": dist_threshold_ratio,
        "figure": fig_path,
        "suitability_notes": suitability_notes
    }
    save_json(summary, os.path.join(output_dir, "watershed_summary.json"))
    return summary


def run_task1(
    photo_path: str = "inputs/photo_original.jpg",
    output_dir: str = "outputs/task1",
    watershed_dist_ratio: float = 0.5,
    watershed_blur: int = 5
) -> Dict[str, Any]:
    """
    Main execution pipeline for Task 1.
    
    Validates input photo, executes restoration filters and segmentation stages,
    and returns comprehensive results dictionary.
    """
    print_banner(
        "TASK 1: IMAGE RESTORATION AND SEGMENTATION",
        f"Input: {photo_path} | Output Dir: {output_dir}"
    )
    
    # Validate input exists and is readable
    validate_photo_original(photo_path, raise_error=True)
    
    # Load image in both RGB (for color restoration & visualization) and Grayscale
    image_rgb = load_rgb(photo_path)
    image_gray = load_gray(photo_path)
    
    h, w, c = image_rgb.shape
    print(f" Loaded '{photo_path}': Dimensions = {w}x{h} px, Channels = {c}")
    
    # Step 1: Noise & Filter Restoration
    restoration_res = run_noise_and_restoration(image_rgb, output_dir=output_dir)
    
    # Step 2: Thresholding & Morphology
    morphology_res = run_thresholding_and_morphology(image_gray, output_dir=output_dir)
    
    # Step 3: Canny Edge Detection
    canny_res = run_canny_edge_detection(image_gray, output_dir=output_dir)
    
    # Step 4: Watershed Segmentation
    watershed_res = run_watershed_segmentation(
        image_rgb,
        output_dir=output_dir,
        dist_threshold_ratio=watershed_dist_ratio,
        blur_ksize=watershed_blur
    )
    
    print("\n" + "=" * 76)
    print(" [COMPLETED] Task 1 pipeline finished successfully!")
    print(f" All outputs and figures saved to: '{output_dir}/'")
    print("=" * 76 + "\n")
    
    return {
        "restoration": restoration_res,
        "morphology": morphology_res,
        "canny": canny_res,
        "watershed": watershed_res
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Task 1: Image Restoration and Segmentation")
    parser.add_argument("--input", default="inputs/photo_original.jpg", help="Path to original photograph")
    parser.add_argument("--output-dir", default="outputs/task1", help="Directory to save task outputs")
    parser.add_argument("--watershed-dist-ratio", type=float, default=0.5, help="Watershed distance threshold ratio")
    parser.add_argument("--watershed-blur", type=int, default=5, help="Kernel size for watershed blur")
    args = parser.parse_args()
    
    run_task1(
        photo_path=args.input,
        output_dir=args.output_dir,
        watershed_dist_ratio=args.watershed_dist_ratio,
        watershed_blur=args.watershed_blur
    )
