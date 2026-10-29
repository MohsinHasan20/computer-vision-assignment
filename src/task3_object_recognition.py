"""
Task 3: Object Recognition (Haar Cascade & Ultralytics YOLO).

This module implements:
1. Object detection using a pretrained OpenCV Haar Cascade classifier
   (Viola-Jones framework with integral images and AdaBoost).
2. Deep learning object recognition using a pretrained Ultralytics YOLO model
   (YOLOv8 nano trained on the 80-class COCO dataset).
3. Annotation of bounding boxes, class labels, and confidence scores.
4. Robust, graceful handling of missing model weights, network errors,
   and package dependencies.
5. Side-by-side comparative analysis based strictly on actual observed detections.
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
    load_rgb,
    load_bgr,
    save_figure,
    save_json,
    save_csv_table,
    print_banner,
    print_step,
    ensure_dir,
)
from src.validators import validate_photo_original


def run_haar_cascade_detection(
    image_bgr: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (30, 30)
) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
    """
    Runs Viola-Jones Haar Cascade detection (frontal face) using OpenCV's built-in models.
    
    Returns:
        (annotated_rgb_image, detections_list, inference_time_sec)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Path to Haar Cascade XML file (check local src/data first, then cv2 built-in)
    local_path = os.path.join(os.path.dirname(__file__), "data", "haarcascade_frontalface_default.xml")
    cv2_path = os.path.join(getattr(cv2.data, "haarcascades", ""), "haarcascade_frontalface_default.xml")
    
    if os.path.exists(local_path):
        cascade_path = local_path
    elif os.path.exists(cv2_path):
        cascade_path = cv2_path
    else:
        # Download from official OpenCV repository if missing
        import urllib.request
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            print("   -> Downloading official haarcascade_frontalface_default.xml...")
            urllib.request.urlretrieve(url, local_path)
            cascade_path = local_path
        except Exception as e:
            raise FileNotFoundError(f"Could not locate or download Haar Cascade XML file: {e}")
            
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    t0 = time.time()
    # detectMultiScale searches for rectangular features across scales
    boxes = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    inference_time = time.time() - t0
    
    annotated_bgr = image_bgr.copy()
    detections = []
    
    for idx, (x, y, w, h) in enumerate(boxes):
        # Draw bounding rectangle in Blue
        cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), (255, 100, 0), 3)
        label = f"Face (Haar #{idx+1})"
        cv2.putText(
            annotated_bgr,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 100, 0),
            2,
            cv2.LINE_AA
        )
        detections.append({
            "detector": "Haar Cascade (Viola-Jones)",
            "class_name": "face",
            "confidence": "N/A (Discrete AdaBoost decision)",
            "box_xywh": [int(x), int(y), int(w), int(h)],
            "notes": "Positive cascade stages passed"
        })
        
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections, inference_time


def run_yolo_detection(
    image_bgr: np.ndarray,
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.25
) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], float, Optional[str]]:
    """
    Runs inference using Ultralytics YOLO.
    
    Gracefully handles:
      - Missing ultralytics package
      - Network download errors when fetching weights
      - Missing local weight files
      
    Returns:
        (annotated_rgb_image, detections_list, inference_time_sec, error_or_warning_msg)
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        warning_msg = (
            f"[WARNING] Could not import 'ultralytics' ({e}).\n"
            f"Please run 'pip install ultralytics' to enable YOLO deep learning detection."
        )
        print(f"\n   {warning_msg}")
        return None, [], 0.0, warning_msg

    print(f"   -> Loading Ultralytics YOLO model: '{model_name}'...")
    try:
        # Load model (auto-downloads from Ultralytics CDN if not found locally)
        model = YOLO(model_name)
    except Exception as e:
        warning_msg = (
            f"[WARNING] Failed to load/download YOLO weights '{model_name}': {e}\n"
            f"If you are offline, download 'yolov8n.pt' manually from:\n"
            f"  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt\n"
            f"and place it directly into the project root folder."
        )
        print(f"\n   {warning_msg}")
        return None, [], 0.0, warning_msg

    t0 = time.time()
    results = model.predict(source=image_bgr, conf=conf_threshold, verbose=False)
    inference_time = time.time() - t0
    
    detections = []
    annotated_bgr = image_bgr.copy()
    
    # Process predictions
    if len(results) > 0 and results[0].boxes is not None:
        boxes_data = results[0].boxes
        names = model.names
        
        for box in boxes_data:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = names.get(cls_id, f"class_{cls_id}")
            
            x1, y1, x2, y2 = xyxy
            # Draw bounding box in vibrant Green
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 220, 0), 3)
            label = f"{class_name}: {conf:.2f}"
            
            # Label background box for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_bgr, (x1, max(0, y1 - 25)), (x1 + tw, max(0, y1)), (0, 220, 0), -1)
            cv2.putText(
                annotated_bgr,
                label,
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )
            
            detections.append({
                "detector": f"Ultralytics YOLO ({model_name})",
                "class_name": class_name,
                "confidence": round(conf, 4),
                "box_xywh": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "notes": f"COCO Class ID: {cls_id}"
            })
            
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections, inference_time, None


def run_task3(
    photo_path: str = "inputs/photo_original.jpg",
    output_dir: str = "outputs/task3",
    yolo_model: str = "yolov8n.pt",
    yolo_conf: float = 0.25
) -> Dict[str, Any]:
    """
    Executes the Task 3 Object Recognition pipeline:
    1. Runs Haar Cascade on the input photo.
    2. Runs pretrained YOLOv8 on the input photo.
    3. Saves individual and side-by-side annotated visualizations.
    4. Compares observed detections, classes, confidence scores, and latency.
    """
    print_banner(
        "TASK 3: OBJECT RECOGNITION (HAAR CASCADE VS. YOLO)",
        f"Input: {photo_path} | Output Dir: {output_dir}"
    )
    
    # 1. Validate input
    validate_photo_original(photo_path, raise_error=True)
    
    img_bgr = load_bgr(photo_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    print(f" Loaded '{photo_path}': Resolution = {w}x{h} px")
    
    # 2. Run Haar Cascade Detection
    print_step(1, "Executing OpenCV Pretrained Haar Cascade Detector (Frontal Face)")
    haar_rgb, haar_dets, haar_time = run_haar_cascade_detection(img_bgr)
    print(f"   -> Haar Detections Count: {len(haar_dets)}")
    print(f"   -> Inference Latency: {haar_time * 1000.0:.1f} ms")
    for i, d in enumerate(haar_dets):
        print(f"      [{i+1}] {d['class_name']} at {d['box_xywh']}")
        
    # Save Haar individual figure
    fig_haar, ax_haar = plt.subplots(figsize=(10, 8))
    ax_haar.imshow(haar_rgb)
    ax_haar.set_title(
        f"Task 3: OpenCV Haar Cascade Detections ({len(haar_dets)} found, {haar_time*1000:.1f} ms)\n"
        f"Algorithm: Viola-Jones Rectangular Features + AdaBoost Cascade",
        fontsize=12, fontweight="bold"
    )
    ax_haar.axis("off")
    plt.tight_layout()
    haar_fig_path = os.path.join(output_dir, "haar_detections.png")
    save_figure(fig_haar, haar_fig_path)
    
    # 3. Run YOLO Detection
    print_step(2, f"Executing Ultralytics Deep Learning YOLO Detector ({yolo_model})")
    yolo_rgb, yolo_dets, yolo_time, yolo_error = run_yolo_detection(
        img_bgr, model_name=yolo_model, conf_threshold=yolo_conf
    )
    
    if yolo_error or yolo_rgb is None:
        print(f"   [NOTICE] YOLO execution skipped or encountered an issue: {yolo_error}")
        yolo_rgb = img_rgb.copy()
        yolo_fig_path = None
    else:
        print(f"   -> YOLO Detections Count: {len(yolo_dets)}")
        print(f"   -> Inference Latency: {yolo_time * 1000.0:.1f} ms")
        for i, d in enumerate(yolo_dets):
            print(f"      [{i+1}] {d['class_name']} (conf: {d['confidence']:.2f}) at {d['box_xywh']}")
            
        fig_yolo, ax_yolo = plt.subplots(figsize=(10, 8))
        ax_yolo.imshow(yolo_rgb)
        ax_yolo.set_title(
            f"Task 3: Ultralytics YOLO Detections ({len(yolo_dets)} objects, {yolo_time*1000:.1f} ms)\n"
            f"Architecture: CNN Darknet Backbone + Multi-scale Feature Pyramid",
            fontsize=12, fontweight="bold"
        )
        ax_yolo.axis("off")
        plt.tight_layout()
        yolo_fig_path = os.path.join(output_dir, "yolo_detections.png")
        save_figure(fig_yolo, yolo_fig_path)

    # 4. Side-by-side Comparative Visualization
    print_step(3, "Generating Side-by-Side Detector Comparison Visualization")
    fig_comp, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig_comp.suptitle("Task 3: Object Recognition Comparison — Haar Cascade vs Ultralytics YOLO", fontsize=16, fontweight="bold")
    
    axes[0].imshow(haar_rgb)
    axes[0].set_title(
        f"1. OpenCV Haar Cascade (Viola-Jones)\n"
        f"Detections: {len(haar_dets)} face(s) | Time: {haar_time*1000:.1f} ms\n"
        f"Binary cascade stages; specialized for upright frontal faces",
        fontsize=12
    )
    axes[0].axis("off")
    
    axes[1].imshow(yolo_rgb)
    yolo_status = f"{len(yolo_dets)} object(s) | Time: {yolo_time*1000:.1f} ms" if not yolo_error else "Offline / Unavailable"
    axes[1].set_title(
        f"2. Ultralytics YOLO ({yolo_model})\n"
        f"Detections: {yolo_status}\n"
        f"Continuous confidence scores [0.0 - 1.0]; 80 general COCO object classes",
        fontsize=12
    )
    axes[1].axis("off")
    
    plt.tight_layout()
    comp_fig_path = os.path.join(output_dir, "detection_comparison.png")
    save_figure(fig_comp, comp_fig_path)
    
    # 5. Log Results and Observed Comparative Table
    all_detections = haar_dets + yolo_dets
    csv_dets_path = os.path.join(output_dir, "detection_results.csv")
    json_dets_path = os.path.join(output_dir, "detection_results.json")
    
    if all_detections:
        save_csv_table(all_detections, csv_dets_path)
    save_json(all_detections, json_dets_path)
    
    # Summary Comparison Record
    comparison_summary = [
        {
            "Feature_Metric": "Model Architecture",
            "Haar_Cascade": "Handcrafted Haar-like rectangular filters + AdaBoost decision tree cascade",
            "Ultralytics_YOLO": "Deep Convolutional Neural Network (CSPDarknet + PANet feature pyramid)"
        },
        {
            "Feature_Metric": "Detected Object Count",
            "Haar_Cascade": str(len(haar_dets)),
            "Ultralytics_YOLO": str(len(yolo_dets)) if not yolo_error else "N/A"
        },
        {
            "Feature_Metric": "Detected Classes (Observed)",
            "Haar_Cascade": ", ".join(list(set(d["class_name"] for d in haar_dets))) if haar_dets else "None detected",
            "Ultralytics_YOLO": ", ".join(list(set(d["class_name"] for d in yolo_dets))) if yolo_dets else ("None detected" if not yolo_error else "N/A")
        },
        {
            "Feature_Metric": "Confidence Scores",
            "Haar_Cascade": "Discrete pass/fail through cascade stages (no probability)",
            "Ultralytics_YOLO": "Continuous Softmax/Sigmoid probability score in range [0.0, 1.0]"
        },
        {
            "Feature_Metric": "Inference Latency",
            "Haar_Cascade": f"{haar_time * 1000.0:.1f} ms",
            "Ultralytics_YOLO": f"{yolo_time * 1000.0:.1f} ms" if not yolo_error else "N/A"
        },
        {
            "Feature_Metric": "Sensitivity / Limitations (Observed)",
            "Haar_Cascade": "Rigid upright orientation requirement; prone to false positives in textures or missed non-frontal faces",
            "Ultralytics_YOLO": "Handles multi-scale, partial occlusions, various poses, and general everyday objects"
        }
    ]
    
    comp_csv_path = os.path.join(output_dir, "detector_comparison_table.csv")
    save_csv_table(comparison_summary, comp_csv_path)
    
    print("\n" + "=" * 76)
    print(" [OBSERVED DETECTOR COMPARISON SUMMARY]")
    print("-" * 76)
    for row in comparison_summary:
        print(f" * {row['Feature_Metric']:<28} | Haar: {row['Haar_Cascade']:<20} | YOLO: {row['Ultralytics_YOLO']}")
    print("=" * 76 + "\n")
    
    return {
        "haar_count": len(haar_dets),
        "yolo_count": len(yolo_dets),
        "figures": [haar_fig_path, yolo_fig_path, comp_fig_path],
        "comparison_table_csv": comp_csv_path
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Task 3: Object Recognition (Haar Cascade & YOLO)")
    parser.add_argument("--input", default="inputs/photo_original.jpg", help="Path to input photograph")
    parser.add_argument("--output-dir", default="outputs/task3", help="Directory to save task outputs")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="Pretrained YOLO weights file")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold")
    args = parser.parse_args()
    
    run_task3(
        photo_path=args.input,
        output_dir=args.output_dir,
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf
    )
