# Computer Vision Individual Practical Assignment

A complete, beginner-friendly Python computer-vision project covering **image restoration, segmentation, feature extraction, object recognition, and video motion estimation**.

> **Crucial Assignment Notice:**  
> This project is designed to run exclusively on **your own unedited personal photos and video clip**. Per assignment guidelines, no stock, generated, or internet images are included or substituted. The project results, measurements, detections, and visual outputs will be produced directly from your own input media once added to `inputs/`.

---

## 📁 Project Structure

```
c:\CCA3 CV\
├── .gitignore                      # Git exclusion rules (media & large weights excluded)
├── README.md                       # Comprehensive project setup and execution guide
├── requirements.txt                # Python package dependencies
├── main.py                         # Unified CLI entry point for all tasks
├── inputs/
│   └── README.md                   # Detailed instructions for taking your personal media
│   # (You add: photo_original.jpg, photo_second.jpg, video_original.mp4)
├── outputs/
│   ├── task1/                     # Filter comparisons, PSNR/SSIM, morphology, Canny, Watershed
│   ├── task2/                     # HOG visualization, SIFT keypoints, matching, invariance tests
│   ├── task3/                     # Haar Cascade & YOLOv8 annotated detections and comparisons
│   └── task4/                     # Differencing, MOG2 background masks, Lucas-Kanade flow
├── report/
│   ├── technical_report_template.md # Complete submission report template with placeholders
│   └── presentation_outline.md    # 5–8 minute video presentation outline and viva guide
└── src/
    ├── __init__.py                # Package initialization
    ├── utils.py                   # Plotting, metric logging, and explanation helpers
    ├── validators.py              # Input media and environment validation checks
    ├── task1_restoration_segmentation.py # Task 1 modular implementation
    ├── task2_features_matching.py # Task 2 modular implementation
    ├── task3_object_recognition.py # Task 3 modular implementation
    └── task4_motion_estimation.py # Task 4 modular implementation
```

---

## 🛠️ Supported Environment & Installation

### Supported Python Versions
* **Python 3.10, 3.11, 3.12, 3.13, or 3.14 (64-bit)**
* Platform: Windows, macOS, or Linux

### 1. (Recommended) Create and Activate a Virtual Environment
Open PowerShell or your terminal inside the project folder:
```powershell
# Create a virtual environment named 'venv'
python -m venv venv

# Activate on Windows PowerShell:
.\venv\Scripts\Activate.ps1
# (Or on Windows Command Prompt: venv\Scripts\activate.bat)
# (Or on macOS/Linux: source venv/bin/activate)
```

### 2. Install Required Dependencies
Install all required libraries using pip:
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Key packages installed:
* `numpy` (array operations and numerical mathematics)
* `opencv-python` (computer vision routines, filtering, morphology, SIFT, Haar, video flow)
* `matplotlib` (publication-quality figure plotting)
* `scikit-image` & `scipy` (PSNR, SSIM, HOG feature extraction)
* `ultralytics` & `torch` (state-of-the-art YOLOv8 deep learning object detection)

---

## 📸 Adding Your Personal Input Media

Before running the tasks, you must place your three personal media files directly into the `inputs/` directory:

| Required File Path | Description | Recommended Specifications |
| :--- | :--- | :--- |
| `inputs/photo_original.jpg` | Your own unedited still photo with clear textures and objects | 1080p+, sharp focus, well-lit (e.g. tabletop items, coins, or portrait) |
| `inputs/photo_second.jpg` | A second photo of the **same subject/scene** | Same subject with 10°–30° angle change or slight scale shift |
| `inputs/video_original.mp4` | 15–30 second video with moving object/person | 720p/1080p at 24–30 FPS; camera stationary/stable for at least part of clip |

> Consult [`inputs/README.md`](inputs/README.md) for detailed photographic tips, lighting advice, and guidelines on selecting suitable items for watershed segmentation.

---

## 🚀 Running the Project

You can run individual tasks or execute the complete pipeline using `main.py`:

### 1. Validate Input Media
Check whether your personal input files are present and readable:
```powershell
python main.py --validate
```
*If any file is missing, the validator will output a clear, actionable guide explaining what is needed.*

### 2. Task 1 — Image Restoration & Segmentation
```powershell
python main.py --task 1
```
* **Operations:** Injects reproducible Gaussian and Salt-and-Pepper noise; applies Gaussian and Median filters; calculates PSNR (dB) and SSIM; performs Otsu binarization, erosion, dilation, opening, and closing; applies multi-stage Canny edge detection; runs marker-controlled Watershed segmentation.
* **Outputs Generated (`outputs/task1/`):**
  * `noise_and_restoration_comparison.png` (2x4 multi-panel comparison)
  * `restoration_metrics.csv` & `restoration_metrics.json` (Numerical PSNR & SSIM table)
  * `morphology_operations.png` (Thresholding and morphological filters)
  * `canny_edge_detection.png` (Edge detection across hysteresis thresholds)
  * `watershed_segmentation.png` (Distance transform, markers, and segmented boundaries)
  * `watershed_summary.json` (Marker count and suitability diagnostic report)

### 3. Task 2 — Feature Extraction & Image Matching
```powershell
python main.py --task 2
```
* **Operations:** Computes dense Histogram of Oriented Gradients (HOG) representations; detects SIFT scale-space keypoints and 128-D descriptors; matches features with Brute-Force Matcher and Lowe's ratio test ($d_1 < 0.75 d_2$); executes a controlled rotation (45°) and scale (0.7x) invariance experiment.
* **Outputs Generated (`outputs/task2/`):**
  * `hog_feature_visualization.png` (Dense gradient orientation maps)
  * `sift_matches_baseline.png` (Correspondences between photo 1 and photo 2)
  * `sift_matches_transformed.png` (Correspondences under 45° rotation and 0.7x scaling)
  * `matching_comparison_summary.png` (Bar chart comparing inlier match counts)
  * `matching_metrics.csv` & `matching_metrics.json` (Match counts, ratios, and summary)

### 4. Task 3 — Object Recognition (Haar Cascade vs. YOLO)
```powershell
python main.py --task 3
```
* **Operations:** Runs OpenCV pretrained Haar Cascade detector (face/upper body) using integral images and AdaBoost; runs Ultralytics YOLOv8 nano model (COCO 80 classes); annotates bounding boxes, classes, and continuous confidence scores; generates an observed comparative evaluation table.
* **Outputs Generated (`outputs/task3/`):**
  * `haar_detections.png` (Viola-Jones detections with bounding boxes)
  * `yolo_detections.png` (YOLOv8 multi-class detections with confidence scores)
  * `detection_comparison.png` (Side-by-side detector comparison visual)
  * `detection_results.csv` & `detection_results.json` (Individual bounding boxes and scores)
  * `detector_comparison_table.csv` (Observed architectural and latency comparison)

### 5. Task 4 — Video Motion Estimation
```powershell
python main.py --task 4
```
* **Operations:** Sequences consecutive video frames; performs two-frame absolute differencing; runs GMM background subtraction (OpenCV MOG2) with dynamic shadow identification; computes Lucas–Kanade pyramidal optical flow with directional motion vectors.
* **Outputs Generated (`outputs/task4/`):**
  * `motion_frame_{N}_comparison.png` (Snapshots showing original frame, diff mask, MOG2 mask, and optical flow vectors)
  * `motion_estimation_composite.png` (Master 4-panel composite figure)
  * `motion_comparison_table.csv` (Observed comparison of latency, FPS, and strengths/trade-offs)
  * `motion_metrics.json` (Video metadata and aggregate motion statistics)

### 6. Run Complete Pipeline (All Tasks End-to-End)
```powershell
python main.py --all
```

---

## 📦 What to Commit to Git vs. What to Upload to Google Drive

To keep the Git repository clean, compliant, and under typical size limits:

| Asset | Submit to Git Repository | Upload to Google Drive / Cloud |
| :--- | :---: | :---: |
| Python source code (`src/*.py`, `main.py`) | ✅ **Yes** | Optional backup |
| Project documentation (`README.md`, `.gitignore`, `requirements.txt`) | ✅ **Yes** | Optional backup |
| Completed Technical Report (`report/technical_report_template.md`) | ✅ **Yes** | ✅ **Yes** (or PDF export) |
| Video Presentation Outline (`report/presentation_outline.md`) | ✅ **Yes** | Optional backup |
| Personal raw video (`inputs/video_original.mp4`) | ❌ **No** (Large file; in `.gitignore`) | ✅ **Yes** (Share via link in report) |
| Personal raw photos (`inputs/photo_original.jpg`, etc.) | ❌ **No** (Kept local / in `.gitignore`) | ✅ **Yes** (Share via link in report) |
| Model weights (`yolov8n.pt`, `*.pt`, `*.pth`) | ❌ **No** (Auto-downloaded; in `.gitignore`) | ❌ No |
| Generated figures and output tables (`outputs/`) | ❌ Kept local (Preserve folders via `.gitkeep`) | ✅ Embed key figures into report |
| 5–8 Minute Explanation Video File (`.mp4`) | ❌ **No** (Too large for Git) | ✅ **Yes** (Paste link into report) |

---

## 🔧 Troubleshooting & Frequently Asked Questions

### 1. `[MISSING INPUT ERROR]` when running any task
* **Cause:** One of the personal media files has not been placed in `inputs/` or is named differently.
* **Fix:** Check spelling and extension: `inputs/photo_original.jpg`, `inputs/photo_second.jpg`, `inputs/video_original.mp4`. Run `python main.py --validate`.

### 2. `cv2.SIFT_create()` raises an error or is unavailable
* **Cause:** Older OpenCV builds or minimal installations.
* **Fix:** The codebase automatically catches this and falls back to ORB while displaying a clear message. To use SIFT directly, ensure `opencv-python>=4.8.0` is installed.

### 3. YOLO fails to download weights or network timeout
* **Cause:** Restricted firewall or offline environment.
* **Fix:** Manually download [`yolov8n.pt`](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt) (approx. 6.2 MB) and save it directly in the project root folder `c:\CCA3 CV\yolov8n.pt`. The script will detect the local file automatically.

### 4. Watershed Segmentation produces too many or zero markers
* **Cause:** Image lacks distinct high-contrast boundaries or has complex high-frequency textures (e.g. hair, carpet).
* **Fix:** Use the CLI parameters to tune the algorithm:
  ```powershell
  python main.py --task 1 --watershed-dist-ratio 0.6 --watershed-blur 7
  ```
  Or consult [`inputs/README.md`](inputs/README.md) for capturing tabletop items with clear background contrast.

### 5. Task 4 Video shows noisy motion across the entire screen
* **Cause:** Camera was handheld and shaking while recording.
* **Fix:** Frame differencing and MOG2 assume a static camera. Prop your phone against a book, mug, or tripod so the camera remains stable while the subject moves.

---

## 📝 Technical Report and Presentation
* The submission report template is located at [`report/technical_report_template.md`](report/technical_report_template.md).
* The 5–8 minute oral video outline and viva preparation script is located at [`report/presentation_outline.md`](report/presentation_outline.md).
