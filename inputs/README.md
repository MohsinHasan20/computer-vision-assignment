# Input Media Requirements & Guidelines

This directory (`inputs/`) is designated for **your own unedited personal photos and video**. 

> **Important Assignment Requirement:**  
> Do **not** use stock photos, internet images, AI-generated pictures, or third-party videos. The assignment requires original, unedited media captured by you. The source code will never modify, overwrite, or delete your original input files.

---

## Required Input Files Checklist

Place the following three files directly into this `inputs/` folder using the exact filenames specified:

| File Path | Description | Recommended Specs | Key Task Usage |
| :--- | :--- | :--- | :--- |
| `inputs/photo_original.jpg` | Your own unedited photograph with visible textures and distinct objects | High resolution (e.g. 1080p+), well-lit, sharp focus | **Task 1** (Restoration, Thresholding, Canny, Watershed) & **Task 3** (Haar, YOLO) |
| `inputs/photo_second.jpg` | A second photograph of the **same object or scene** taken from a slightly different angle or distance | Same subject, slight perspective or scale variation | **Task 2** (HOG visualization, SIFT keypoints & descriptor matching) |
| `inputs/video_original.mp4` | Your own unedited **15–30 second video** showing a moving person or object against a textured background | 1080p or 720p at 24–30 FPS; camera stationary/stable for at least a 5-second segment | **Task 4** (Frame differencing, MOG2 background subtraction, Lucas–Kanade optical flow) |

---

## Detailed Guidelines for Capturing Your Media

### 1. `inputs/photo_original.jpg` (Primary Photograph)
* **What to capture:**
  * An everyday tabletop scene with multiple distinct items (for example: coins, keys, fruits, small bottles, caps, or stationery on a desk) **OR** a portrait photo of yourself/a friend with background objects.
* **Why this matters for Task 1 (Restoration & Segmentation):**
  * **Noise & Filters:** Detailed textures allow you to observe how Gaussian smoothing reduces noise while blurring edges, and how median filtering cleanly eliminates salt-and-pepper impulses without destroying sharp transitions.
  * **Canny Edge Detection:** Sharp edges and strong contrast yield clear, continuous contours.
  * **Watershed Segmentation:** Watershed works best on images containing distinct, foreground objects resting on a contrasting background. If objects touch slightly (e.g. adjacent coins or buttons), the distance transform can separate their regional peaks and allow watershed markers to segment individual instances.
* **What makes a photo unsuitable for Watershed?**
  * Flat, textureless backgrounds without contrast (e.g. pure white paper with white items).
  * Extreme clutter where everything overlaps completely into a single unidentifiable mass.
  * Non-uniform harsh shadows that get falsely detected as objects during thresholding.
  * *Tip:* If your primary photo is a human portrait, the watershed algorithm may over-segment hair or clothing folds. The code includes adjustable threshold settings, but for best watershed results, capturing distinct objects on a surface is ideal.

### 2. `inputs/photo_second.jpg` (Matching Pair)
* **What to capture:**
  * The exact same subject or scene as `photo_original.jpg`, but captured with:
    * A slight shift in viewpoint (10°–30° angle difference).
    * A small change in distance (closer or further away).
    * Natural lighting variation.
* **Why this matters for Task 2 (Feature Extraction & SIFT Matching):**
  * SIFT (Scale-Invariant Feature Transform) is designed to be invariant to image scale and rotation, and robust to modest viewpoint and illumination variations. Having two real-world photos of the same physical scene tests true feature matching.
  * The script also performs a controlled synthetic rotation/scaling experiment to quantitatively prove invariance.

### 3. `inputs/video_original.mp4` (Motion Video)
* **What to capture:**
  * 15 to 30 seconds of footage.
  * **Crucial:** Keep the camera **as still as possible** (prop it on a table, bookshelf, or use a tripod/two hands braced against a surface) for at least part of the clip.
  * Have a person walk across the frame, a pet move, or a hand place/move an object across the table.
  * Ensure the background has some visible detail (not a plain blank wall).
* **Why this matters for Task 4 (Video Motion Estimation):**
  * **Frame Differencing:** Calculates $|I_t - I_{t-1}|$. If the camera shakes, every background pixel moves, generating false motion noise. A stable camera ensures only genuine moving foreground items are highlighted.
  * **GMM Background Subtraction (MOG2):** Builds a Gaussian Mixture Model of the static background. It requires a few frames where the background stays in place to learn the stationary distribution.
  * **Lucas–Kanade Optical Flow:** Assumes brightness constancy ($I(x,y,t) = I(x+dx, y+dy, t+dt)$) and spatial coherence. Visible textures provide the corner/gradient features required by `cv2.goodFeaturesToTrack()`.

---

## Verification & Error Prevention

Before running the pipeline, check:
1. Files are placed directly inside `inputs/` (not in a sub-folder).
2. The extensions match: `.jpg` (or `.jpeg`) for images, `.mp4` for video.
3. You can verify that all inputs are recognized by running:
   ```bash
   python main.py --validate
   ```
   If any file is missing or unreadable, the command will display a helpful, descriptive message pointing out what is missing.
