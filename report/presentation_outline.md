# 5–8 Minute Project Explanation Video Outline & Script Guide

> **Important Reminders for Your Video Presentation:**
> 1. **Explain the work in your own words.** Assessors want to verify your individual comprehension of the concepts, algorithms, and practical trade-offs. Avoid reading raw text off the screen.
> 2. **Show your own actual input media and generated outputs.** Demonstrate your photographs and video clip.
> 3. **Keep an active pace:** Aim for approximately **6 to 7 minutes** total (safely within the 5–8 minute window).
> 4. **Recording tools:** You can use OBS Studio, Zoom (recording a meeting with yourself), Loom, or Microsoft PowerPoint Record Screen. Ensure your microphone audio is crisp and your terminal / figures are zoomed in for readability.

---

## Minute-by-Minute Presentation Breakdown

```
[0:00 - 1:00]  Introduction, Project Purpose & Personal Input Media
[1:00 - 2:30]  Task 1: Image Restoration, Metrics (PSNR/SSIM) & Segmentation
[2:30 - 3:45]  Task 2: Feature Extraction (HOG) & SIFT Invariance Matching
[3:45 - 5:00]  Task 3: Object Recognition (Haar Cascade vs. Ultralytics YOLO)
[5:00 - 6:30]  Task 4: Video Motion Estimation (Differencing, MOG2, Lucas–Kanade)
[6:30 - 7:30]  Conclusion, Limitations & Key Learnings
```

---

### Segment 1: Introduction & Personal Input Media (0:00 – 1:00)
* **What to show on screen:**
  * Title slide or your terminal with project structure.
  * Your three original personal media files: `photo_original.jpg`, `photo_second.jpg`, and a 3-second preview of `video_original.mp4`.
* **What to say (in your own words):**
  * *"Hello, my name is [Your Name], and this is my individual computer vision assignment covering image restoration, feature matching, object recognition, and video motion estimation."*
  * *"To satisfy the assignment rules, all experiments are run on my own original, unedited media. Here is my primary photograph showing [mention the objects in your photo, e.g. coins, keys, or desk items], a second photograph taken from a different angle to test feature matching, and a 20-second video clip with a moving subject filmed with a stable camera."*
  * *"Let's look at Task 1: Image Restoration and Segmentation."*

---

### Segment 2: Task 1 — Restoration, PSNR/SSIM & Segmentation (1:00 – 2:30)
* **What to show on screen:**
  * `outputs/task1/noise_and_restoration_comparison.png`
  * `outputs/task1/morphology_operations.png`
  * `outputs/task1/watershed_segmentation.png`
* **What to say (in your own words):**
  * **Noise & Filters:**
    * *"For image restoration, I injected two controlled noise models: zero-mean Gaussian noise simulating sensor thermal noise, and Salt-and-Pepper impulse noise simulating bit errors."*
    * *"I then applied Gaussian spatial smoothing and non-linear median filtering to both."*
  * **PSNR & SSIM Explanation:**
    * *"To evaluate restoration quality objectively, I calculated PSNR and SSIM. PSNR measures absolute numerical fidelity in decibels where higher is better, while SSIM measures perceptual structural similarity on a 0 to 1 scale."*
    * *"As you can observe in my results table: on Gaussian noise, the Gaussian filter improved PSNR to [X] dB, but softened sharp boundaries. On Salt-and-Pepper noise, the Gaussian filter failed completely because it averaged impulse spikes into gray smudges (PSNR only [Y] dB). In contrast, the median filter was optimal—it completely stripped the impulse outliers, restoring sharp edges and achieving a high SSIM of [Z]."*
  * **Segmentation (Canny & Watershed):**
    * *"For segmentation, I applied Otsu’s thresholding followed by morphological opening to clean noise and closing to bridge gaps."*
    * *"Canny edge detection was applied using hysteresis thresholding to extract continuous contours."*
    * *"Finally, for watershed segmentation, I computed the Euclidean distance transform to identify sure foreground peaks, labeled connected components, and used watershed flooding to segment touching objects along boundary ridge lines."*

---

### Segment 3: Task 2 — HOG Features & SIFT Matching (2:30 – 3:45)
* **What to show on screen:**
  * `outputs/task2/hog_feature_visualization.png`
  * `outputs/task2/sift_matches_baseline.png`
  * `outputs/task2/sift_matches_transformed.png`
* **What to say (in your own words):**
  * **HOG Representation:**
    * *"In Task 2, I explored feature representations. First, I computed Histogram of Oriented Gradients (HOG), which captures local edge directions over 16x16 pixel blocks. The visualization shows how HOG forms a compact structural signature of the scene."*
  * **SIFT Keypoints & Lowe's Ratio Test:**
    * *"Next, I used SIFT to detect scale-space extrema and extract 128-dimensional local gradient descriptors."*
    * *"To match descriptors between my first and second photograph, I used a Brute-Force Matcher paired with Lowe's ratio test (threshold 0.75). This filters out ambiguous matches where the second-nearest neighbor is almost as close as the nearest."*
  * **Invariance Experiment:**
    * *"To rigorously test rotation and scale invariance, I rotated the second image by 45 degrees and scaled it down to 70%. As shown in the comparison, SIFT successfully maintained [N] valid correspondences even under rotation and scaling, confirming its mathematical invariance properties."*

---

### Segment 4: Task 3 — Object Recognition (3:45 – 5:00)
* **What to show on screen:**
  * `outputs/task3/detection_comparison.png`
  * `outputs/task3/detector_comparison_table.csv` (or table in the report)
* **What to say (in your own words):**
  * **Architectural Comparison:**
    * *"In Task 3, I compared two distinct eras of object recognition: the classical Viola-Jones Haar Cascade detector versus a modern deep learning YOLOv8 neural network."*
    * *"The Haar cascade relies on handcrafted rectangular features computed rapidly with integral images and filtered through an AdaBoost cascade. YOLOv8 is a single-stage deep convolutional neural network trained on the 80-class COCO dataset."*
  * **Observed Results:**
    * *"Looking at my actual output: Haar cascade executed in [X] ms and detected [Y] face(s), but provides only a binary detection decision without confidence probabilities, and cannot recognize general objects."*
    * *"YOLOv8 took [A] ms and detected [B] objects—including [mention objects detected: e.g. person with 0.92 confidence, cup with 0.85]. YOLO provides continuous probability confidences and robustly handles varied poses and background clutter."*

---

### Segment 5: Task 4 — Video Motion Estimation (5:00 – 6:30)
* **What to show on screen:**
  * `outputs/task4/motion_estimation_composite.png`
  * `outputs/task4/motion_comparison_table.csv`
* **What to say (in your own words):**
  * **Method Walkthrough:**
    * *"In Task 4, I analyzed motion in my supplied video clip using three different computer vision techniques:"*
    * *"1. Two-Frame Differencing calculates the absolute difference between consecutive frames. It is computationally fast (averaging [X] ms per frame), but exhibits 'ghosting' artifacts, leaves hollow interiors, and completely loses tracking if the subject pauses."*
    * *"2. GMM Background Subtraction (OpenCV MOG2) models every pixel over time using a mixture of Gaussians. It adapts to gradual lighting changes, distinguishes moving foreground from stationary background, and isolates shadows. It produced complete, solid silhouettes."*
    * *"3. Lucas-Kanade Optical Flow solves the brightness constancy constraint to compute exact 2D velocity vectors. Here you can see the cyan motion arrows and magenta tracked corner points showing the exact direction and speed of movement."*
  * **Comparative Assessment:**
    * *"Rather than declaring an arbitrary 'winner', my comparison table shows that each method serves a distinct purpose: Differencing for low-power trigger sensors, MOG2 for silhouette segmentation, and Lucas-Kanade for trajectory and velocity estimation."*

---

### Segment 6: Conclusion & Viva Preparation (6:30 – 7:30)
* **What to show on screen:**
  * Summary slide or conclusion section of your report.
* **What to say (in your own words):**
  * *"In conclusion, this project demonstrated how classic digital signal processing, invariant feature engineering, and modern deep neural networks solve real-world vision challenges."*
  * *"Some practical limitations I encountered included tuning watershed distance ratios to avoid over-segmentation on textured surfaces, and the sensitivity of optical flow to sudden rapid displacements."*
  * *"All source code is organized modularly in my Git repository, and reproducible instructions are documented in the README. Thank you for watching!"*

---

## Anticipated Viva / Defense Questions & Quick Answers

1. **Q: Why does the median filter outperform the Gaussian filter on salt-and-pepper noise?**  
   *A:* Gaussian filtering is linear averaging; it blurs extreme 0 and 255 spikes into gray patches without eliminating them. Median filtering is a rank-order non-linear operation; because impulse spikes are extreme statistical outliers, they are discarded by taking the median value of the neighborhood.

2. **Q: What is the purpose of Lowe's ratio test in SIFT matching?**  
   *A:* If the closest descriptor distance $d_1$ is almost equal to the second-closest distance $d_2$ ($d_1 / d_2 \approx 1.0$), the feature match is ambiguous (e.g. repeated patterns or noise). Requiring $d_1 < 0.75 \cdot d_2$ filters out false and ambiguous matches.

3. **Q: Why does two-frame differencing create hollow silhouettes?**  
   *A:* If the interior of an object has uniform color or texture, the pixel values inside the object do not change significantly between consecutive frames. Only the moving boundaries/edges create intensity differences.

4. **Q: What is the optical flow brightness constancy assumption?**  
   *A:* It assumes that the intensity of a physical surface point does not change as it moves between frames: $I(x, y, t) = I(x + \Delta x, y + \Delta y, t + \Delta t)$.
