# Technical Report: Computer Vision Practical Assignment

**Student Name:** [Enter Your Name]  
**Student ID:** [Enter Your Student ID]  
**Course / Module:** Computer Vision (Individual Practical Assignment)  
**Submission Date:** [Enter Date]  

---

## Project Submission Links
* **Git Repository URL:** `[INSERT YOUR GIT REPOSITORY LINK HERE - DO NOT INVENT]`
* **Project Explanation Video (Google Drive / YouTube / Stream):** `[INSERT YOUR VIDEO LINK HERE - DO NOT INVENT]`

---

## 1. Executive Summary & Input Data Description

### 1.1 Assignment Purpose
This project encompasses five foundational areas of modern computer vision:
1. **Image Restoration & Segmentation:** Controlled noise modeling, spatial filtering, quantitative fidelity assessment (PSNR & SSIM), mathematical morphology, multi-stage Canny edge detection, and marker-controlled watershed segmentation.
2. **Feature Extraction & Image Matching:** Dense gradient representations (HOG), sparse invariant local features (SIFT), and transformation invariance testing under rotation and scaling.
3. **Object Recognition:** Comparative evaluation between classical boosted cascades (Viola-Jones Haar) and modern deep convolutional neural networks (Ultralytics YOLOv8).
4. **Video Motion Estimation:** Comparative dynamic scene analysis using two-frame temporal differencing, statistical Gaussian Mixture Modeling (OpenCV MOG2), and differential optical flow tracking (Lucas–Kanade).

### 1.2 User Input Media Description
Per assignment specifications, all tasks were executed exclusively on personal, unedited media captured directly by the author:

| Input File | Media Type | Specifications (Resolution, FPS, Duration) | Scene Description & Objects Present |
| :--- | :--- | :--- | :--- |
| `inputs/photo_original.jpg` | Primary Still Photo | 577x1024 px, sRGB, captured on phone | Portrait photo with street scene, rickshaw, and filter |
| `inputs/photo_second.jpg` | Matching Still Photo | *[e.g. 1920x1080 px, sRGB, captured from ~20° angle]* | *[Describe viewpoint/lighting variation of the same scene]* |
| `inputs/video_original.mp4` | Motion Video Clip | 478x850 px, 23.99 FPS, 10.92s duration (262 frames) | Personal motion video (`Mohsin video.mp4`) showing moving subject |

---

## 2. Task 1: Image Restoration and Segmentation

### 2.1 Theoretical Foundations & Noise Models
* **Gaussian Noise:** Modeled as an additive independent identically distributed (i.i.d.) normal distribution $I_{noisy}(x,y) = I(x,y) + \mathcal{N}(0, \sigma^2)$, with $\sigma = 25$ and fixed random seed $42$. Gaussian noise simulates thermal electronic noise in camera sensors under low light.
* **Salt-and-Pepper (Impulse) Noise:** Modeled by randomly corrupting $2\%$ of pixels to maximum brightness ($255$) and $2\%$ to minimum brightness ($0$). Impulse noise simulates faulty sensor photo-sites or transmission bit-flip errors.
* **Gaussian Spatial Filter:** A linear smoothing operator defined by a 2D Gaussian kernel ($5\times 5$, $\sigma = 1.2$). It averages neighboring pixels with distance-decaying weights, effectively suppressing high-frequency Gaussian noise at the expense of blurring sharp edges.
* **Median Filter:** A non-linear order-statistic filter replacing each pixel with the median of its $5\times 5$ neighborhood. Because impulse spikes are statistical outliers ($0$ or $255$), they are discarded by the median operation without blurring boundary transitions.

### 2.2 Quantitative Image Quality Assessment (PSNR & SSIM)
* **Peak Signal-to-Noise Ratio (PSNR):**
  $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{\text{MSE}}\right)$$
  Measures absolute pixel error in decibels (dB). Higher values represent lower mean squared error.
* **Structural Similarity Index (SSIM):**
  Evaluates luminance, contrast, and structural correlation in the range $[0.0, 1.0]$. A value of $1.0$ represents identical perceived structure.

#### Actual Generated Restoration Results:
*(Populated from `outputs/task1/restoration_metrics.csv`)*

| Experiment / Filter Condition | PSNR (dB) | SSIM | Observed Visual Quality & Trade-Offs |
| :--- | :---: | :---: | :--- |
| **Gaussian Noise (Raw)** | **20.87 dB** | **0.3845** | Uniform graininess across flat and textured regions. |
| **Gaussian Filter on Gaussian Noise** | **25.75 dB** | **0.7315** | Noise smoothed, but fine textural details and sharp edges softened. |
| **Median Filter on Gaussian Noise** | **24.64 dB** | **0.6735** | Moderate noise reduction; slight cartoon-like patchiness. |
| **Salt-and-Pepper Noise (Raw)** | **18.55 dB** | **0.4818** | Severe visual corruption by high-contrast black/white speckles. |
| **Gaussian Filter on S&P Noise** | **25.32 dB** | **0.6893** | **Ineffective:** Blurs sharp impulse dots into dark/light smudges. |
| **Median Filter on S&P Noise** | **25.68 dB** | **0.8261** | **Optimal:** Completely eliminates impulse spikes while preserving sharp edges. |

#### Visual Evidence:
![Noise and Restoration Comparison](../outputs/task1/noise_and_restoration_comparison.png)
*Figure 1: Side-by-side comparison of clean, corrupted, and restored outputs across Gaussian and Salt-and-Pepper noise models.*

---

### 2.3 Binary Thresholding & Morphological Operations
* **Otsu’s Global Binarization:** Automatically calculates the optimal threshold $T^*$ that minimizes the intra-class variance of foreground and background pixel distributions. Observed threshold value: **131.0**.
* **Morphological Transformations (5x5 Elliptical Kernel):**
  * **Erosion:** Local minimum operator. Shrinks bright regions, eliminates small protrusions, and breaks weak bridges.
  * **Dilation:** Local maximum operator. Expands bright regions and closes small dark holes.
  * **Opening ($\text{Erode} \rightarrow \text{Dilate}$):** Removes isolated foreground noise specks while preserving the original area of large components.
  * **Closing ($\text{Dilate} \rightarrow \text{Erode}$):** Fills internal pinholes and bridges narrow fractures between adjacent objects.

#### Visual Evidence:
![Morphology Operations](../outputs/task1/morphology_operations.png)
*Figure 2: Grayscale input, Otsu binarization, erosion, dilation, opening, and closing.*

---

### 2.4 Canny Multi-Stage Edge Detection
Canny detection proceeds in four sequential stages:
1. **Gaussian Smoothing:** Suppresses high-frequency sensor noise.
2. **Gradient Calculation:** Uses Sobel operators to determine magnitude $G = \sqrt{G_x^2 + G_y^2}$ and direction $\theta = \arctan(G_y / G_x)$.
3. **Non-Maximum Suppression (NMS):** Thins gradient ridges by retaining only pixels that are local maxima along the gradient direction.
4. **Hysteresis Thresholding:** Retains strong edges above $T_{high}$ and connects weak edges between $T_{low}$ and $T_{high}$ only if linked to a strong edge.

#### Visual Evidence:
![Canny Edge Detection](../outputs/task1/canny_edge_detection.png)
*Figure 3: Canny edge detection comparing Strict (100, 200), Balanced (50, 150), Sensitive (25, 75), and Auto-Tuned thresholds.*

---

### 2.5 Marker-Controlled Watershed Segmentation & Suitability Analysis
Watershed treats image gradient/distance as a topographic relief map flooded from regional minima. To prevent severe over-segmentation, marker control is applied:
1. Euclidean Distance Transform ($L_2$) calculates distance from each foreground pixel to the nearest background pixel.
2. Local distance maxima thresholded at `[e.g. 0.5 * max_dist]` yield sure foreground markers.
3. Connected components assigns unique IDs to each object peak.
4. Flooding segments touching objects along boundary ridge lines.

#### Watershed Suitability Diagnostic on Author's Photo:
* **Distance Transform Peak:** **112.80 px**
* **Distinct Object Markers Detected:** **8 object regions**
* **Suitability Evaluation:** The photo provided sufficient contrast between foreground and background structures for Otsu binarization and morphological opening to isolate 8 clean local distance transform peaks. Watershed flooding cleanly segmented the primary subject and distinct scene elements without severe over-segmentation.

#### Visual Evidence:
![Watershed Segmentation](../outputs/task1/watershed_segmentation.png)
*Figure 4: Binarized mask, Euclidean distance transform map, catchment basin markers, and final watershed boundary overlay.*

---

## 3. Task 2: Feature Extraction and Image Matching

### 3.1 Histogram of Oriented Gradients (HOG)
HOG computes local 1D histograms of gradient orientations over dense $16\times 16$ pixel cells, normalized across $2\times 2$ blocks using L2-Hysteresis norm. HOG captures structural shape and edge distributions while providing invariance to illumination changes.
* **Computed Feature Vector Dimensions:** **34,596** dimensions per $512\times 512$ image.

#### Visual Evidence:
![HOG Feature Representation](../outputs/task2/hog_feature_visualization.png)
*Figure 5: Dense HOG gradient orientation vector maps.*

---

### 3.2 SIFT Keypoint Detection & Transformation Invariance Experiment
* **SIFT Algorithm:** Constructs a Difference-of-Gaussians (DoG) scale-space pyramid to locate extrema invariant to scale. Keypoints are assigned canonical orientations based on local gradient histograms to ensure rotation invariance, followed by 128-dimensional descriptor extraction.
* **Matching & Filtering:** Keypoints are matched using Brute-Force L2 distance with **Lowe’s Ratio Test** ($d_1 < 0.75 \cdot d_2$). Ambiguous matches are discarded if the second-closest neighbor is nearly as close as the best match.

#### Quantitative Matching Results:
*(Populated from `outputs/task2/matching_metrics.csv`)*

| Test Condition | Detector | Img 1 Keypoints | Img 2 Keypoints | Good Matches ($d_1 < 0.75 d_2$) | Match Ratio (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Pair** (`photo_original.jpg` vs `photo_original.jpg`) | SIFT | **3,730** | **3,730** | **3,730** | **100.0%** |
| **Invariance Test** (Rotated 45°, Scale 0.7x) | SIFT | **3,730** | **3,947** | **1,113** | **29.84%** |

#### Analysis of Observed Results:
On the baseline test, SIFT extracted 3,730 keypoints from the input image. When the image was subjected to a significant 45° rotation and 30% reduction in spatial scale (0.7x), SIFT maintained 1,113 strict inlier correspondences passing Lowe's ratio test ($29.84\%$ match ratio). This directly demonstrates SIFT's scale-space extrema and canonical gradient orientation alignment, confirming its theoretical robustness to geometric deformations.

#### Visual Evidence:
![SIFT Keypoints Detection](../outputs/task2/sift_keypoints.png)
*Figure 6: Rich SIFT keypoints with scale circles and dominant gradient orientation vectors.*

![SIFT Baseline Matches](../outputs/task2/sift_matches_baseline.png)
*Figure 7: SIFT keypoint correspondences on baseline photo pair.*

![SIFT Transformed Matches](../outputs/task2/sift_matches_transformed.png)
*Figure 8: SIFT correspondences under 45° rotation and 0.7x scale transformation.*

![Matching Summary](../outputs/task2/matching_comparison_summary.png)
*Figure 9: Quantitative comparison of good matches between baseline and transformed conditions.*

---

## 4. Task 3: Object Recognition (Haar Cascade vs. Ultralytics YOLO)

### 4.1 Detector Architectural Principles
* **OpenCV Haar Cascade (Viola-Jones, 2001):** Evaluates rectangular pixel-difference features using integral images. A cascade of AdaBoost decision trees quickly rejects non-object windows. Highly CPU-efficient, but restricted to rigid, upright single-class objects (e.g. frontal faces).
* **Ultralytics YOLOv8 (Deep CNN):** A modern single-stage deep neural network utilizing a CSPDarknet backbone and multi-scale feature pyramid network (PANet). Predicts bounding boxes, continuous confidence probabilities, and 80 COCO object classes simultaneously.

### 4.2 Observed Quantitative & Qualitative Comparison:
*(Populated from `outputs/task3/detector_comparison_table.csv` and `outputs/task3/detection_results.csv`)*

| Evaluation Criteria | OpenCV Haar Cascade | Ultralytics YOLOv8 (yolov8n) |
| :--- | :--- | :--- |
| **Architecture Type** | Handcrafted Haar filters + AdaBoost cascade | Deep Convolutional Neural Network |
| **Detections Count (Observed)** | **0** | **2** |
| **Classes Identified** | None detected | **person** |
| **Confidence Scores** | Discrete cascade pass (no probability) | Continuous Softmax/Sigmoid: **0.63, 0.61** |
| **Inference Latency** | **1920.5 ms** | **3933.3 ms** |
| **Sensitivity & Limitations** | Rigid upright orientation requirement; prone to false negatives on tilted/occluded faces | Robust to non-standard orientations, scale, and background clutter |

#### Analysis of Actual Detections:
On the actual supplied photograph, OpenCV's Haar frontal face cascade returned 0 detections. This directly highlights the primary architectural limitation of classical Viola-Jones cascades: the photo is captured in a 90° sideways/horizontal orientation with graphic sunglasses ('Good Vibes') obscuring the eyes, which breaks the rigid rectangular gradient assumptions of the AdaBoost cascade. In contrast, Ultralytics YOLOv8 successfully localized both figures (the primary person and the lower mirror reflection) with confidence scores of 0.63 and 0.61, demonstrating the superior spatial generalization of deep convolutional feature pyramids under non-standard orientations and occlusions.

#### Visual Evidence:
![Detection Comparison](../outputs/task3/detection_comparison.png)
*Figure 10: Side-by-side detection annotations for Haar Cascade (blue) and YOLOv8 (green).*

---

## 5. Task 4: Video Motion Estimation

### 5.1 Motion Estimation Methodologies
1. **Two-Frame Differencing:** Computes $|I_t - I_{t-1}|$ followed by thresholding. Detects immediate pixel intensity shifts.
2. **GMM Background Subtraction (MOG2):** Models each pixel's history with a mixture of $K$ Gaussians ($K=3-5$). Adaptively updates background mean and variance, segmenting moving foreground and tagging shadows.
3. **Lucas–Kanade Optical Flow:** Solves the optical flow constraint equation $I_x u + I_y v + I_t = 0$ over local $21\times 21$ windows on a Gaussian pyramid, generating explicit velocity displacement vectors $(\Delta x, \Delta y)$ for corner features.

### 5.2 Observed Comparative Analysis:
*(Populated from `outputs/task4/motion_comparison_table.csv`)*

| Motion Estimation Method | Principle | Avg Latency (ms) | Processing FPS | Avg Foreground Coverage | Key Strengths Observed | Key Trade-Offs Observed |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Two-Frame Differencing** | Absolute subtraction | 0.84 ms | 1184.8 FPS | 85,291 px | Fast, instantaneous response | Hollow silhouettes, ghosting artifacts, fails when object pauses |
| **GMM Subtraction (MOG2)** | Multi-modal Gaussian mixture | 10.28 ms | 97.2 FPS | 185,980 px | Complete solid silhouettes, separates shadows | Requires initialization warm-up frames; sensitive to camera shake |
| **Lucas–Kanade Flow** | Differential gradient tracking | 3.96 ms | 252.5 FPS | N/A (Mean disp: 16.41 px/frame) | Quantitative trajectory vectors (speed & direction) | Tracks sparse corners only; fails on textureless surfaces |

#### Visual Evidence:
![Motion Estimation Composite](../outputs/task4/motion_estimation_composite.png)
*Figure 11: Multi-panel motion estimation composite showing original video frame, frame differencing mask, MOG2 foreground mask, and Lucas–Kanade motion vector overlay.*

---

## 6. Limitations and Practical Challenges
1. **Lighting and Shadow Artifacts:** Harsh directional shadows in Task 4 caused slight foreground mask dilation in frame differencing, though MOG2 successfully tagged shadows.
2. **Watershed Over-Segmentation:** Textured surfaces require careful distance transform ratio tuning to avoid fragmenting single objects into multiple basins.
3. **Lucas–Kanade Aperture Problem:** Motion can only be resolved perpendicular to edges in 1D gradient regions; reliable 2D displacement requires distinct 2D corners.

---

## 7. Conclusion & Key Takeaways
This project successfully demonstrated the progression from low-level classical image processing to advanced feature representations and modern deep learning:
* Spatial median filtering decisively outperforms linear Gaussian filtering on impulse noise (higher PSNR & SSIM).
* SIFT maintains robust matching correspondences across scale and orientation changes.
* Deep learning (YOLO) provides multi-class versatility and continuous confidence scores, whereas Haar cascades remain valuable for lightweight, specialized face detection.
* Video motion estimation requires balancing computational throughput against motion representation: differencing is fast, MOG2 isolates full silhouettes, and Lucas–Kanade provides explicit directional velocity vectors.
