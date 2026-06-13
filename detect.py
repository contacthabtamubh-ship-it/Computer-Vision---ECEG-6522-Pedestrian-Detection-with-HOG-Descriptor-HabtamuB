#!/usr/bin/env python3
"""
Pedestrian Detection using HOG + SVM
Works with INRIA dataset structure:
    ~/pedestrian_detection_project/data/INRIA_Person_Dataset/Test/JPEGImages/
"""

import cv2
import numpy as np
import os
import glob
import argparse
import time
from typing import List, Tuple, Optional

# ------------------------------------------------------------
# 1. HOG Descriptor (pre-trained on INRIA)
# ------------------------------------------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# ------------------------------------------------------------
# 2. Non-Maximum Suppression
# ------------------------------------------------------------
def non_max_suppression(
    boxes: np.ndarray, weights: List[float], iou_threshold: float = 0.3
) -> Tuple[np.ndarray, List[float]]:
    if len(boxes) == 0:
        return np.array([]), []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(weights)[::-1]
    keep = []

    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[indices[1:]] - intersection)
        indices = indices[1:][iou < iou_threshold]

    return boxes[keep].astype(int), [weights[i] for i in keep]


# ------------------------------------------------------------
# 3. Detect on a single image
# ------------------------------------------------------------
def detect_pedestrians(
    image_path: str,
    output_path: Optional[str] = None,
    win_stride: Tuple[int, int] = (8, 8),
    padding: Tuple[int, int] = (16, 16),
    scale: float = 1.05,
) -> Tuple[Optional[np.ndarray], Optional[List[float]], float]:
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read {image_path}")
        return None, None, 0.0

    start = time.time()
    boxes, weights = hog.detectMultiScale(img, winStride=win_stride, padding=padding, scale=scale)
    elapsed = time.time() - start

    boxes, weights = non_max_suppression(boxes, weights)

    if output_path and len(boxes) > 0:
        result = img.copy()
        for (x, y, w, h), conf in zip(boxes, weights):
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, f"{conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(output_path, result)
        print(f"  Saved: {output_path}")

    return boxes, weights, elapsed


# ------------------------------------------------------------
# 4. Process a whole folder
# ------------------------------------------------------------
def process_folder(image_folder: str, output_folder: str, limit: int = 0) -> None:
    if not os.path.isdir(image_folder):
        print(f"ERROR: Folder not found: {image_folder}")
        return

    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(image_folder, ext)))
    images = sorted(images)

    if not images:
        print(f"No images found in {image_folder}")
        return

    if limit > 0:
        images = images[:limit]

    os.makedirs(output_folder, exist_ok=True)
    total_time = 0.0
    print(f"\nProcessing {len(images)} images...")
    for i, img_path in enumerate(images, 1):
        base = os.path.basename(img_path)
        out_path = os.path.join(output_folder, f"detected_{base}")
        _, _, t = detect_pedestrians(img_path, out_path)
        total_time += t
        print(f"[{i}/{len(images)}] {base} done in {t:.3f}s")

    if len(images) > 0:
        print(f"\nAverage time per image: {total_time / len(images):.3f}s")


# ------------------------------------------------------------
# 5. Evaluate detection rate (recall)
# ------------------------------------------------------------
def evaluate_detection_rate(image_folder: str) -> None:
    if not os.path.isdir(image_folder):
        print(f"ERROR: Folder not found: {image_folder}")
        return

    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(image_folder, ext)))
    images = sorted(images)

    if not images:
        print(f"No images found in {image_folder}")
        return

    detected = 0
    total_time = 0.0
    print(f"\nEvaluating {len(images)} images...")
    for img_path in images:
        boxes, _, t = detect_pedestrians(img_path, None)
        total_time += t
        if boxes is not None and len(boxes) > 0:
            detected += 1

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total test images:        {len(images)}")
    print(f"Images with detections:   {detected}")
    print(f"Detection rate (Recall):  {detected / len(images) * 100:.2f}%")
    print(f"Average processing time:  {total_time / len(images):.3f} seconds per image")
    print("=" * 50)


# ------------------------------------------------------------
# 6. Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pedestrian Detection HOG+SVM")
    parser.add_argument(
        "--mode",
        choices=["evaluate", "process"],
        required=True,
        help="'evaluate' = detection rate, 'process' = save annotated images"
    )
    parser.add_argument(
        "--image_folder",
        required=True,
        help="Path to folder with images (e.g., .../Test/JPEGImages) or a single image file"
    )
    parser.add_argument(
        "--output_folder",
        default="./detection_output",
        help="Folder to save annotated images (only for process mode)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max images to process (0 = all, only for process mode)"
    )

    args = parser.parse_args()

    if args.mode == "evaluate":
        evaluate_detection_rate(args.image_folder)
    else:  # process
        if os.path.isfile(args.image_folder):
            os.makedirs(args.output_folder, exist_ok=True)
            base = os.path.basename(args.image_folder)
            out_path = os.path.join(args.output_folder, f"detected_{base}")
            detect_pedestrians(args.image_folder, out_path)
        else:
            process_folder(args.image_folder, args.output_folder, args.limit)


if __name__ == "__main__":
    main()