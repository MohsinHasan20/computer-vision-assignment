"""
Input Validation Module for Computer Vision Project.

Validates that user-provided personal photos and video exist, are readable,
and meet basic dimensional/formatting requirements. Produces clear, friendly,
and actionable error messages for students.
"""

import os
import sys
from typing import Optional, Tuple
import cv2


class InputValidationError(Exception):
    """Custom exception raised when an input file is missing or invalid."""
    pass


def validate_photo_original(
    file_path: str = "inputs/photo_original.jpg",
    raise_error: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validates existence and readability of inputs/photo_original.jpg.
    
    Args:
        file_path: Path to the primary photograph.
        raise_error: If True, raises InputValidationError on failure.
        
    Returns:
        (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        msg = (
            f"\n"
            f"======================================================================\n"
            f" [MISSING INPUT ERROR]: Primary photograph not found!\n"
            f" Expected file location: '{file_path}'\n"
            f"======================================================================\n"
            f" To run this task, you must provide your own unedited personal photo.\n"
            f" \n"
            f" Quick Action Required:\n"
            f"   1. Take a well-lit photograph containing clear textures and distinct\n"
            f"      objects (e.g. coins/keys on a desk, or a portrait/scene).\n"
            f"   2. Save or copy it directly into the 'inputs/' folder with the name:\n"
            f"      -> '{file_path}'\n"
            f"   3. Note: Per assignment rules, do not use stock or downloaded photos.\n"
            f"   4. Consult 'inputs/README.md' for capture tips and watershed advice.\n"
            f"======================================================================\n"
        )
        if raise_error:
            raise InputValidationError(msg)
        return False, msg

    img = cv2.imread(file_path)
    if img is None:
        msg = (
            f"\n"
            f"======================================================================\n"
            f" [UNREADABLE IMAGE ERROR]: Cannot decode '{file_path}'!\n"
            f"======================================================================\n"
            f" OpenCV was unable to decode this image. Possible causes:\n"
            f"   - The file is empty (0 bytes) or corrupted.\n"
            f"   - The file was renamed to .jpg from an incompatible format (e.g. HEIC/WEBP).\n"
            f" Solution:\n"
            f"   Convert your photo to standard JPEG format (sRGB color space)\n"
            f"   and re-save as '{file_path}'.\n"
            f"======================================================================\n"
        )
        if raise_error:
            raise InputValidationError(msg)
        return False, msg

    h, w, c = img.shape
    if h < 100 or w < 100:
        msg = (
            f"[WARNING] Image '{file_path}' is very small ({w}x{h} px). "
            f"A higher resolution photograph (e.g. 1080p+) is recommended for clear features."
        )
        return True, msg

    return True, None


def validate_photo_second(
    file_path: str = "inputs/photo_second.jpg",
    raise_error: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validates existence and readability of inputs/photo_second.jpg (for Task 2 matching).
    """
    if not os.path.exists(file_path):
        msg = (
            f"\n"
            f"======================================================================\n"
            f" [MISSING INPUT ERROR]: Second photograph not found!\n"
            f" Expected file location: '{file_path}'\n"
            f"======================================================================\n"
            f" Task 2 requires two photos of the SAME subject/scene to test\n"
            f" SIFT feature matching under viewpoint or lighting variations.\n"
            f" \n"
            f" Quick Action Required:\n"
            f"   1. Take a second photograph of the exact same subject as 'photo_original.jpg'\n"
            f"      from a slightly different angle (10-30 deg) or distance.\n"
            f"   2. Place it into the 'inputs/' folder with the exact filename:\n"
            f"      -> '{file_path}'\n"
            f"   3. Refer to 'inputs/README.md' for matching guidelines.\n"
            f"======================================================================\n"
        )
        if raise_error:
            raise InputValidationError(msg)
        return False, msg

    img = cv2.imread(file_path)
    if img is None:
        msg = (
            f"\n"
            f"======================================================================\n"
            f" [UNREADABLE IMAGE ERROR]: Cannot decode '{file_path}'!\n"
            f" Ensure the photo is valid JPEG and not an unreadable raw/HEIC format.\n"
            f"======================================================================\n"
        )
        if raise_error:
            raise InputValidationError(msg)
        return False, msg

    return True, None


def validate_video_original(
    file_path: str = "inputs/video_original.mp4",
    raise_error: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validates existence, openability, and duration/frames of inputs/video_original.mp4.
    """
    if not os.path.exists(file_path):
        msg = (
            f"\n"
            f"======================================================================\n"
            f" [MISSING INPUT ERROR]: Video file not found!\n"
            f" Expected file location: '{file_path}'\n"
            f"======================================================================\n"
            f" Task 4 requires your own 15–30 second unedited video clip showing\n"
            f" a moving person or object with background detail and stable camera.\n"
            f" \n"
            f" Quick Action Required:\n"
            f"   1. Record a 15–30 second video with your phone/camera.\n"
            f"      Keep the camera steady on a table or tripod for part of the clip\n"
            f"      while someone/something moves across the frame.\n"
            f"   2. Save or copy it into the 'inputs/' folder as:\n"
            f"      -> '{file_path}'\n"
            f"   3. Consult 'inputs/README.md' for motion estimation tips.\n"
            f"======================================================================\n"
        )
        if raise_error:
            raise InputValidationError(msg)
        return False, msg

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        msg = (
            f"\n"
            f"======================================================================\n"
            f" [UNREADABLE VIDEO ERROR]: OpenCV cannot open '{file_path}'!\n"
            f" Check that the video codec is supported (H.264 / MP4 recommended).\n"
            f"======================================================================\n"
        )
        if raise_error:
            raise InputValidationError(msg)
        return False, msg

    ret, frame = cap.read()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not ret or frame is None or frame_count <= 0:
        msg = (
            f"[ERROR] Could not read any valid video frames from '{file_path}'. "
            f"The video file appears empty or incomplete."
        )
        if raise_error:
            raise InputValidationError(msg)
        return False, msg

    return True, None


def validate_all_inputs(verbose: bool = True) -> bool:
    """
    Validates all required inputs and prints a clear summary table.
    Does not raise exceptions; returns True if all present, False otherwise.
    """
    results = {}
    v1, msg1 = validate_photo_original(raise_error=False)
    results["inputs/photo_original.jpg"] = (v1, msg1, "Task 1 (Restoration/Segmentation) & Task 3 (Recognition)")

    v2, msg2 = validate_photo_second(raise_error=False)
    results["inputs/photo_second.jpg"] = (v2, msg2, "Task 2 (Feature Extraction & Matching)")

    v3, msg3 = validate_video_original(raise_error=False)
    results["inputs/video_original.mp4"] = (v3, msg3, "Task 4 (Video Motion Estimation)")

    all_present = all(v[0] for v in results.values())

    if verbose:
        print("\n" + "=" * 76)
        print(" INPUT MEDIA VALIDATION STATUS CHECK")
        print("=" * 76)
        for filepath, (is_valid, msg, tasks) in results.items():
            status_tag = "[OK - FOUND]" if is_valid else "[MISSING]"
            print(f" {status_tag:<14} : {filepath}")
            print(f"   Required for : {tasks}")
            if not is_valid:
                print(f"   Status       : Not detected in project workspace.")
            else:
                print(f"   Status       : Verified readable.")
        print("=" * 76)
        if not all_present:
            print(" Summary: One or more personal input files are currently missing.")
            print(" Please see 'inputs/README.md' for guidelines on adding your own media.")
            print("=" * 76 + "\n")
        else:
            print(" Summary: All required input media files are present and verified!\n" + "=" * 76 + "\n")

    return all_present
