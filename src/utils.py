"""
Utility Functions for Computer Vision Project.

Provides helper routines for:
- Directory creation
- Safe image loading and color conversion
- Matplotlib plot styling and artifact saving
- JSON and CSV metric logging
- Student-friendly console logging and metric explanations
"""

import os
import json
import csv
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> str:
    """Ensures that a directory exists, creating parent folders if necessary."""
    os.makedirs(path, exist_ok=True)
    return path


def load_rgb(image_path: str) -> np.ndarray:
    """
    Loads an image from disk and converts from OpenCV BGR to standard RGB.
    
    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Failed to load image from '{image_path}'. Ensure file exists and is valid.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_bgr(image_path: str) -> np.ndarray:
    """Loads an image directly in OpenCV BGR format."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Failed to load image from '{image_path}'. Ensure file exists and is valid.")
    return bgr


def load_gray(image_path: str) -> np.ndarray:
    """Loads an image directly in single-channel grayscale format."""
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Failed to load image from '{image_path}'. Ensure file exists and is valid.")
    return gray


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 300) -> str:
    """
    Saves a Matplotlib figure with high resolution and tight bounding box,
    then closes the figure to free memory.
    """
    ensure_dir(os.path.dirname(filepath))
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"   [SAVED FIGURE] -> {filepath}")
    return filepath


def save_json(data: Any, filepath: str) -> str:
    """Saves structured metrics or results to a nicely formatted JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"   [SAVED METRICS] -> {filepath}")
    return filepath


def save_csv_table(records: List[Dict[str, Any]], filepath: str, fieldnames: Optional[List[str]] = None) -> str:
    """Saves a list of dictionary records to a CSV table."""
    if not records:
        return filepath
    ensure_dir(os.path.dirname(filepath))
    if fieldnames is None:
        fieldnames = list(records[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"   [SAVED CSV TABLE] -> {filepath}")
    return filepath


def print_banner(title: str, subtitle: Optional[str] = None):
    """Prints a styled banner for tasks or major steps."""
    bar = "=" * 76
    print(f"\n{bar}")
    print(f"  {title.upper()}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"{bar}\n")


def print_step(step_idx: int, description: str):
    """Prints a numbered step indicator."""
    print(f"\n>> Step {step_idx}: {description}")
    print("-" * 60)


def print_metric_explanation():
    """
    Prints a clear educational explanation of PSNR and SSIM for oral defense.
    """
    print("""
----------------------------------------------------------------------
[UNDERSTANDING PSNR AND SSIM - KEY CONCEPTS FOR ORAL EXPLANATION]
----------------------------------------------------------------------
1. PSNR (Peak Signal-to-Noise Ratio):
   - Formula: PSNR = 10 * log10( (MAX_I)^2 / MSE )
   - Measured in: Decibels (dB)
   - Interpretation:
     * Measures absolute numerical fidelity between original and restored image.
     * Higher is BETTER: A higher PSNR means lower Mean Squared Error (MSE).
     * Typical range for 8-bit images:
       - < 20 dB : Poor quality, heavy noise or severe blur visible.
       - 20-30 dB: Acceptable quality, noticeable softening or residual artifacts.
       - 30-40 dB: Good to excellent quality, near-imperceptible distortion.
       - > 40 dB : Extremely high quality, visually indistinguishable.

2. SSIM (Structural Similarity Index Measure):
   - Range: [-1.0, 1.0] (Practically [0.0, 1.0] for image comparisons)
   - Interpretation:
     * Evaluates visual perception based on luminance, contrast, and structural correlation.
     * Higher is BETTER: 1.0 indicates perfect structural identity with the clean reference.
     * PSNR can sometimes be misled by uniform blurring, whereas SSIM closely aligns
       with human visual system (HVS) perception.
----------------------------------------------------------------------
""")
