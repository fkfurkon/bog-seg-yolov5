#!/usr/bin/env python3
"""
Enhanced Instance Segmentation Inference with Advanced Visualization.

This script performs comprehensive instance segmentation inference with:
- YOLOv5 segmentation model
- Combined bounding box + mask visualization
- Thumbnail grid generation
- Detailed analysis output
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add YOLOv5 imports
sys.path.append(".")
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.segment.general import process_mask, scale_masks
from utils.torch_utils import select_device


class EnhancedInstanceSegmentation:
    def __init__(self, weights_path, device="cpu", conf_thres=0.25, iou_thres=0.45):
        """Initialize enhanced instance segmentation."""
        self.device = select_device(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Load model
        print(f"Loading model: {weights_path}")
        self.model = DetectMultiBackend(weights_path, device=self.device)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt

        # Get image size
        self.img_size = check_img_size(640, s=self.stride)

        # Food class colors (vibrant colors for better visualization)
        self.class_colors = {
            0: (255, 100, 100),  # protein - red
            1: (255, 200, 100),  # carbohydrate - orange
            2: (100, 255, 100),  # fruit - green
            3: (255, 100, 255),  # dessert - magenta
            4: (200, 200, 200),  # flatware - gray
            5: (100, 255, 200),  # vegetable - light green
            6: (200, 100, 255),  # sauce - purple
            7: (255, 255, 100),  # soup - yellow
            8: (100, 200, 255),  # snack - light blue
        }

        print("✅ Model loaded successfully")
        print(f"📱 Device: {self.device}")
        print(f"🏷️  Classes: {list(self.names.values())}")

    def preprocess_image(self, img_path):
        """Preprocess image for inference."""
        # Read image
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            raise ValueError(f"Image not found: {img_path}")

        # Letterbox
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.pt)[0]

        # Convert HWC to CHW, BGR to RGB
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0

        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        return img, img0

    def run_inference(self, img):
        """Run inference."""
        with torch.no_grad():
            pred, proto = self.model(img, augment=False, visualize=False)[:2]
        return pred, proto

    def postprocess_predictions(self, pred, proto, img, img0):
        """Post-process predictions."""
        # Apply NMS
        pred = non_max_suppression(
            pred,
            self.conf_thres,
            self.iou_thres,
            nc=len(self.names),
            agnostic=False,
            max_det=1000,
            nm=32,  # number of mask coefficients
        )

        results = []

        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

                # Process masks
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)
                masks = masks.cpu().numpy()

                # Scale masks to original image size
                masks = scale_masks(img.shape[2:], masks, img0.shape)

                # Extract data
                boxes = det[:, :4].cpu().numpy()
                scores = det[:, 4].cpu().numpy()
                classes = det[:, 5].cpu().numpy().astype(int)

                results.append({"boxes": boxes, "scores": scores, "classes": classes, "masks": masks})
            else:
                results.append(
                    {"boxes": np.array([]), "scores": np.array([]), "classes": np.array([]), "masks": np.array([])}
                )

        return results

    def create_advanced_visualization(self, img0, results, save_path=None):
        """Create advanced visualization with both boxes and masks."""
        if len(results) == 0 or len(results[0]["boxes"]) == 0:
            return img0

        result = results[0]
        boxes = result["boxes"]
        scores = result["scores"]
        classes = result["classes"]
        masks = result["masks"]

        # Create overlay image
        overlay = img0.copy()

        # Draw masks first (underneath boxes)
        for i in range(len(boxes)):
            if i < len(masks):
                mask = masks[i]
                class_id = classes[i]
                color = self.class_colors.get(class_id, (128, 128, 128))

                # Create colored mask
                colored_mask = np.zeros_like(img0, dtype=np.uint8)
                colored_mask[mask > 0.5] = color

                # Blend mask with image (semi-transparent)
                overlay = cv2.addWeighted(overlay, 0.75, colored_mask, 0.25, 0)

        # Draw bounding boxes and labels
        for i in range(len(boxes)):
            box = boxes[i]
            score = scores[i]
            class_id = classes[i]
            class_name = self.names[class_id]
            color = self.class_colors.get(class_id, (128, 128, 128))

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)

            # Prepare label
            label = f"{class_name} {score:.2f}"

            # Get text size
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Draw label background
            cv2.rectangle(overlay, (x1, y1 - text_height - baseline - 10), (x1 + text_width + 10, y1), color, -1)

            # Draw label text
            cv2.putText(
                overlay, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
            )

        if save_path:
            cv2.imwrite(str(save_path), overlay)
            print(f"💾 Visualization saved: {save_path}")

        return overlay

    def predict_and_visualize(self, img_path, save_path=None):
        """Complete prediction and visualization pipeline."""
        print(f"🔍 Processing: {Path(img_path).name}")

        # Preprocess
        img, img0 = self.preprocess_image(img_path)

        # Inference
        pred, proto = self.run_inference(img)

        # Postprocess
        results = self.postprocess_predictions(pred, proto, img, img0)

        # Visualize
        result_img = self.create_advanced_visualization(img0, results, save_path)

        # Print summary
        if len(results[0]["boxes"]) > 0:
            print(f"  ✅ Found {len(results[0]['boxes'])} objects:")
            for i, (box, score, class_id) in enumerate(
                zip(results[0]["boxes"], results[0]["scores"], results[0]["classes"])
            ):
                class_name = self.names[class_id]
                print(f"    {i + 1}. {class_name}: {score:.3f}")
        else:
            print("  ⚠️  No objects detected")

        return result_img, results


def create_comprehensive_thumbnail_grid(segmentation_engine, image_paths, output_path):
    """Create comprehensive thumbnail grid with detailed annotations."""
    # Create figure with larger size for better detail
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(
        "Food Instance Segmentation Results\nบทรวจจับและแบ่งส่วนอาหาร (Bounding Boxes + Segmentation Masks)",
        fontsize=18,
        fontweight="bold",
        y=0.95,
    )

    # Process each image
    for i, img_path in enumerate(image_paths[:4]):
        row = i // 2
        col = i % 2

        print(f"\n📸 Processing thumbnail {i + 1}: {Path(img_path).name}")

        # Run inference
        result_img, results = segmentation_engine.predict_and_visualize(img_path)

        # Convert BGR to RGB for matplotlib
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # Display image
        axes[row, col].imshow(result_img_rgb)
        axes[row, col].set_title(
            f"รูปที่ {i + 1}: {Path(img_path).name}\nจำนวนวัตถุที่ตรวจพบ: {len(results[0]['boxes'])} objects",
            fontsize=14,
            fontweight="bold",
        )
        axes[row, col].axis("off")

        # Add detection summary as text
        if len(results[0]["boxes"]) > 0:
            detection_text = "ตรวจพบ:\n"
            class_counts = {}
            for class_id in results[0]["classes"]:
                class_name = segmentation_engine.names[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            for class_name, count in class_counts.items():
                detection_text += f"• {class_name}: {count}\n"

            # Add text box with detection summary
            axes[row, col].text(
                0.02,
                0.98,
                detection_text,
                transform=axes[row, col].transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
            )

    # Add legend for classes
    legend_text = "คลาสอาหาร (Food Classes):\n"
    for class_id, class_name in segmentation_engine.names.items():
        np.array(segmentation_engine.class_colors.get(class_id, (128, 128, 128))) / 255
        legend_text += f"• {class_name}\n"

    # Add overall legend
    fig.text(
        0.02, 0.02, legend_text, fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8)
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.15)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"🖼️  Comprehensive thumbnail grid saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Instance Segmentation Inference")
    parser.add_argument(
        "--weights", default="runs/train/food_instance_segmentation/weights/best.pt", help="Path to trained weights"
    )
    parser.add_argument("--source", default="yolo_dataset_final/images/val", help="Source directory or image file")
    parser.add_argument("--output", default="runs/inference/enhanced_instance_segmentation", help="Output directory")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--device", default="cpu", help="Device to run inference")
    parser.add_argument("--max-images", type=int, default=8, help="Maximum images to process")

    args = parser.parse_args()

    print("=" * 80)
    print("🎯 ENHANCED FOOD INSTANCE SEGMENTATION")
    print("=" * 80)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check weights
    if not os.path.exists(args.weights):
        print(f"❌ Weights not found: {args.weights}")
        print("Please train the model first using train_tensorboard_segmentation.py")
        return

    # Initialize segmentation engine
    segmentation_engine = EnhancedInstanceSegmentation(
        weights_path=args.weights, device=args.device, conf_thres=args.conf_thres, iou_thres=args.iou_thres
    )

    # Get image paths
    source_path = Path(args.source)
    if source_path.is_file():
        image_paths = [source_path]
    else:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        image_paths = []
        for ext in extensions:
            image_paths.extend(source_path.glob(f"*{ext}"))
            image_paths.extend(source_path.glob(f"*{ext.upper()}"))

    if not image_paths:
        print(f"❌ No images found in: {source_path}")
        return

    # Limit and sort images
    image_paths = sorted(image_paths)[: args.max_images]
    print(f"📊 Processing {len(image_paths)} images...")

    # Process individual images
    print(f"\n{'=' * 60}")
    print("📸 PROCESSING INDIVIDUAL IMAGES")
    print(f"{'=' * 60}")

    for i, img_path in enumerate(image_paths):
        output_path = output_dir / f"result_{img_path.stem}.jpg"
        result_img, results = segmentation_engine.predict_and_visualize(img_path, output_path)

    # Create comprehensive thumbnail grid for first 4 images
    if len(image_paths) >= 4:
        print(f"\n{'=' * 60}")
        print("🖼️  CREATING COMPREHENSIVE THUMBNAIL GRID")
        print(f"{'=' * 60}")

        thumbnail_path = output_dir / "comprehensive_thumbnail_grid.jpg"
        create_comprehensive_thumbnail_grid(segmentation_engine, image_paths[:4], thumbnail_path)

    # Summary
    print(f"\n{'=' * 80}")
    print("✅ INFERENCE COMPLETED SUCCESSFULLY!")
    print(f"{'=' * 80}")
    print(f"📁 Results directory: {output_dir}")
    print(f"📊 Individual results: {output_dir}/result_*.jpg")
    if len(image_paths) >= 4:
        print(f"🖼️  Thumbnail grid: {output_dir}/comprehensive_thumbnail_grid.jpg")
    print(f"🎯 Processed {len(image_paths)} images with {len(segmentation_engine.names)} food classes")


if __name__ == "__main__":
    main()
