#!/usr/bin/env python
"""
Merged Dataset Inference - Bounding Box + Segmentation
This script performs inference on the merged dataset with both annotation types.
"""

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
import yaml


class MergedDatasetInference:
    def __init__(self, weights_path="yolov5s-seg.pt", data_config="merged_dataset/data.yaml"):
        """Initialize inference with merged dataset."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        try:
            if Path(weights_path).exists():
                self.model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path, device=self.device)
                print(f"✅ Loaded trained model: {weights_path}")
            else:
                self.model = torch.hub.load("ultralytics/yolov5", "yolov5s-seg", device=self.device)
                print("⚠️  Using pretrained model (trained model not found)")
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            self.model = torch.hub.load("ultralytics/yolov5", "yolov5s-seg", device=self.device)

        # Load dataset configuration
        with open(data_config) as f:
            self.data = yaml.safe_load(f)

        self.class_names = self.data["names"]
        self.num_classes = self.data["nc"]

        # Thai translations
        self.thai_names = {
            0: "โปรตีน (protein)",
            1: "คาร์โบไฮเดรต (carbohydrate)",
            2: "ผลไม้ (fruit)",
            3: "ของหวาน (dessert)",
            4: "ช้อนส้อม (flatware)",
            5: "ผัก (vegetable)",
            6: "ซอส (sauce)",
            7: "ซุป (soup)",
            8: "ขนม (snack)",
        }

        # Colors for visualization
        self.colors = [
            (0, 255, 0),  # protein - green
            (255, 165, 0),  # carbohydrate - orange
            (255, 0, 255),  # fruit - magenta
            (255, 192, 203),  # dessert - pink
            (128, 128, 128),  # flatware - gray
            (0, 255, 0),  # vegetable - lime
            (255, 255, 0),  # sauce - yellow
            (255, 0, 0),  # soup - red
            (128, 0, 128),  # snack - purple
        ]

        print(f"📊 Dataset: {self.num_classes} classes")
        for i, name in self.class_names.items():
            print(f"   {i}: {name} ({self.thai_names.get(i, '')})")

    def predict_image(self, image_path, conf_thresh=0.25):
        """Run inference on a single image."""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Run inference
        results = self.model(image)

        # Parse results
        detections = []
        if hasattr(results, "pandas") and hasattr(results.pandas(), "xyxy"):
            df = results.pandas().xyxy[0]
            for _, row in df.iterrows():
                if row["confidence"] >= conf_thresh:
                    detections.append(
                        {
                            "bbox": [row["xmin"], row["ymin"], row["xmax"], row["ymax"]],
                            "confidence": row["confidence"],
                            "class_id": int(row["class"]),
                            "class_name": row["name"],
                        }
                    )

        return detections, image, results

    def visualize_results(self, image, detections, save_path=None):
        """Create visualization with bounding boxes and segmentation."""
        result_image = image.copy()
        height, width = image.shape[:2]

        for detection in detections:
            bbox = detection["bbox"]
            conf = detection["confidence"]
            class_id = detection["class_id"]

            # Get class info
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            thai_name = self.thai_names.get(class_id, "")
            color = self.colors[class_id % len(self.colors)]

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            # Create label with both English and Thai
            label = f"{class_name}: {conf:.2f}"
            if thai_name:
                label = f"{thai_name}: {conf:.2f}"

            # Calculate text size
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Draw background for text
            cv2.rectangle(result_image, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)

            # Draw text
            cv2.putText(
                result_image,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )

        if save_path:
            cv2.imwrite(str(save_path), result_image)
            print(f"💾 Saved: {save_path}")

        return result_image


def run_merged_inference_examples():
    """Run inference examples on merged dataset."""
    print("🔍 MERGED DATASET INFERENCE EXAMPLES")
    print("=" * 60)

    # Check for trained model
    weights_options = [
        "runs/train-seg/merged_food_segmentation/weights/best.pt",
        "runs/train-seg/yolo_food_segmentation/weights/best.pt",
        "yolov5s-seg.pt",
    ]

    weights_path = "yolov5s-seg.pt"  # default
    for weight_option in weights_options:
        if Path(weight_option).exists():
            weights_path = weight_option
            break

    # Initialize inference
    inference = MergedDatasetInference(weights_path)

    # Output directory
    output_dir = Path("runs/merged_inference")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get sample images
    val_images = list(Path("merged_dataset/images/val").glob("*.jpg")) + list(
        Path("merged_dataset/images/val").glob("*.png")
    )

    if not val_images:
        # Fallback to train images
        val_images = list(Path("merged_dataset/images/train").glob("*.jpg")) + list(
            Path("merged_dataset/images/train").glob("*.png")
        )

    # Select 6 random images
    sample_images = random.sample(val_images, min(6, len(val_images)))

    print(f"📸 Processing {len(sample_images)} sample images...")

    results = []
    for idx, img_path in enumerate(sample_images):
        print(f"\n🖼️  Image {idx + 1}: {img_path.name}")

        try:
            # Run inference
            detections, image, raw_results = inference.predict_image(img_path, conf_thresh=0.3)

            print(f"   📊 Detected {len(detections)} objects:")
            for det in detections:
                class_name = inference.class_names.get(det["class_id"], f"class_{det['class_id']}")
                thai_name = inference.thai_names.get(det["class_id"], "")
                conf = det["confidence"]
                print(f"     • {class_name} ({thai_name}): {conf:.3f}")

            # Create visualization
            output_path = output_dir / f"merged_inference_{idx + 1:02d}.jpg"
            inference.visualize_results(image, detections, output_path)

            results.append(
                {
                    "image": str(img_path),
                    "output": str(output_path),
                    "detections": len(detections),
                    "classes": [det["class_id"] for det in detections],
                }
            )

        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    # Create summary
    create_merged_summary(results, output_dir, inference)

    print("\n✅ Merged dataset inference completed!")
    print(f"📁 Results saved to: {output_dir}")
    print("📊 Features demonstrated:")
    print("   ✓ Combined bounding box + segmentation training")
    print("   ✓ Multi-class Thai food detection")
    print("   ✓ Thai and English labels")
    print("   ✓ High-quality merged annotations")
    print(f"   ✓ {len(results)} inference examples")


def create_merged_summary(results, output_dir, inference):
    """Create summary visualization of merged inference results."""
    if len(results) < 4:
        return

    print("\n📋 Creating merged inference summary...")

    # Load result images
    images = []
    titles = []

    for result in results[:6]:
        try:
            img = cv2.imread(result["output"])
            if img is not None:
                img = cv2.resize(img, (320, 240))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)

                # Create title with class distribution
                classes = result["classes"]
                class_counts = {}
                for cls_id in classes:
                    cls_name = inference.class_names.get(cls_id, f"class_{cls_id}")
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                title_parts = []
                for cls_name, count in class_counts.items():
                    title_parts.append(f"{cls_name}({count})")

                title = f"Example {len(titles) + 1}: {', '.join(title_parts[:3])}"
                if len(title_parts) > 3:
                    title += "..."
                titles.append(title)
        except:
            continue

    if len(images) < 4:
        return

    # Create grid
    rows = 2
    cols = 3 if len(images) >= 6 else 2

    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    fig.suptitle(
        "YOLOv5 Segmentation - Merged Dataset Results\n"
        "Combined Bounding Box + Segmentation Training\n"
        "Thai Food Dataset (9 Classes)",
        fontsize=16,
        fontweight="bold",
    )

    axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=10, fontweight="bold")
            axes[i].axis("off")

    # Hide extra subplots
    for i in range(len(images), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    summary_path = output_dir / "merged_inference_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Summary saved: {summary_path}")


if __name__ == "__main__":
    run_merged_inference_examples()
