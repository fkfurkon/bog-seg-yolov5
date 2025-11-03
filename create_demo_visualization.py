#!/usr/bin/env python
"""
Simple Instance Segmentation Visualization
Creates thumbnail visualizations with bounding boxes and segmentation masks overlay.
"""

import random
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_class_names(data_file):
    """Load class names from dataset yaml file."""
    with open(data_file) as f:
        data = yaml.safe_load(f)
    return data["names"]


def generate_colors(num_classes):
    """Generate distinct colors for each class."""
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = plt.cm.hsv(hue)[:3]
        colors.append([int(c * 255) for c in rgb])
    return colors


def create_demo_visualization():
    """Create demo visualization with sample images and fake predictions."""
    # Load class names
    class_names = load_class_names("food_dataset.yaml")
    colors = generate_colors(len(class_names))

    # Get some sample images
    image_dir = Path("dataset/image")
    image_files = list(image_dir.glob("*.png"))[:4]  # Get first 4 images

    if len(image_files) < 4:
        image_files = list(image_dir.glob("*.jpg"))[:4]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle("Instance Segmentation Demo - Food Dataset", fontsize=16, fontweight="bold")

    axes = axes.flatten()

    for idx, img_path in enumerate(image_files[:4]):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Create demo predictions (random for visualization)
        num_objects = random.randint(2, 5)

        # Create overlay image
        overlay = img_rgb.copy()

        for obj_idx in range(num_objects):
            # Random bounding box
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)
            x2 = random.randint(x1 + 50, min(x1 + 200, w))
            y2 = random.randint(y1 + 50, min(y1 + 200, h))

            # Random class
            class_idx = random.randint(0, len(class_names) - 1)
            class_name = class_names[class_idx]
            color = colors[class_idx]

            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)

            # Create fake segmentation mask
            mask = np.zeros((h, w), dtype=np.uint8)
            # Create elliptical mask within bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            axis_a = (x2 - x1) // 3
            axis_b = (y2 - y1) // 3
            cv2.ellipse(mask, (center_x, center_y), (axis_a, axis_b), 0, 0, 360, 255, -1)

            # Apply mask with transparency
            mask_colored = np.zeros_like(img_rgb)
            mask_colored[mask > 0] = color
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

            # Add label
            conf = random.uniform(0.7, 0.95)
            label = f"{class_name} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(overlay, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display
        axes[idx].imshow(overlay)
        axes[idx].set_title(f"Sample {idx + 1}: {img_path.name}")
        axes[idx].axis("off")

    # Add legend
    legend_elements = []
    for i, name in enumerate(class_names.values()):
        color = [c / 255.0 for c in colors[i]]  # Normalize for matplotlib
        legend_elements.append(patches.Patch(color=color, label=name))

    fig.legend(handles=legend_elements, loc="center", bbox_to_anchor=(0.5, 0.02), ncol=len(class_names))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save visualization
    output_dir = Path("runs/demo_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "instance_segmentation_demo.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Demo visualization saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_demo_visualization()
