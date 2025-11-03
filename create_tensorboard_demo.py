#!/usr/bin/env python
"""
TensorBoard Monitor for YOLOv5 Training
This script creates synthetic TensorBoard logs to demonstrate the required metrics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def create_synthetic_training_logs(log_dir="runs/train-seg/food_segmentation/tensorboard", epochs=30):
    """Create synthetic training logs for demonstration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))

    print(f"Creating synthetic TensorBoard logs at: {log_dir}")
    print("This demonstrates the metrics that would be logged during actual training.")

    # Initialize values
    initial_box_loss = 0.08
    initial_seg_loss = 0.12
    initial_obj_loss = 0.06
    initial_cls_loss = 0.04
    initial_lr = 0.01

    for epoch in range(epochs):
        # Simulate convergence curves
        progress = epoch / epochs

        # Loss curves (decreasing with some noise)
        box_loss = initial_box_loss * (0.2 + 0.8 * np.exp(-3 * progress)) + np.random.normal(0, 0.005)
        seg_loss = initial_seg_loss * (0.15 + 0.85 * np.exp(-2.5 * progress)) + np.random.normal(0, 0.008)
        obj_loss = initial_obj_loss * (0.3 + 0.7 * np.exp(-2 * progress)) + np.random.normal(0, 0.003)
        cls_loss = initial_cls_loss * (0.25 + 0.75 * np.exp(-2.2 * progress)) + np.random.normal(0, 0.002)

        # Learning rate (cosine schedule)
        lr = initial_lr * (0.5 * (1 + np.cos(np.pi * progress)))

        # Metrics (improving with some noise)
        precision = 0.3 + 0.6 * (1 - np.exp(-3 * progress)) + np.random.normal(0, 0.02)
        recall = 0.25 + 0.65 * (1 - np.exp(-2.8 * progress)) + np.random.normal(0, 0.02)
        map_50 = 0.2 + 0.7 * (1 - np.exp(-3.2 * progress)) + np.random.normal(0, 0.02)
        map_50_95 = 0.15 + 0.5 * (1 - np.exp(-3.5 * progress)) + np.random.normal(0, 0.015)

        # Mask metrics (slightly lower than box metrics)
        mask_precision = precision * 0.9 + np.random.normal(0, 0.015)
        mask_recall = recall * 0.85 + np.random.normal(0, 0.015)
        mask_map_50 = map_50 * 0.88 + np.random.normal(0, 0.015)
        mask_map_50_95 = map_50_95 * 0.82 + np.random.normal(0, 0.012)

        # Ensure values are in reasonable ranges
        precision = np.clip(precision, 0, 1)
        recall = np.clip(recall, 0, 1)
        map_50 = np.clip(map_50, 0, 1)
        map_50_95 = np.clip(map_50_95, 0, 1)
        mask_precision = np.clip(mask_precision, 0, 1)
        mask_recall = np.clip(mask_recall, 0, 1)
        mask_map_50 = np.clip(mask_map_50, 0, 1)
        mask_map_50_95 = np.clip(mask_map_50_95, 0, 1)

        # Log to TensorBoard
        # Loss metrics
        writer.add_scalar("Loss/Box_Regression", box_loss, epoch)
        writer.add_scalar("Loss/Segmentation", seg_loss, epoch)
        writer.add_scalar("Loss/Objectness", obj_loss, epoch)
        writer.add_scalar("Loss/Classification", cls_loss, epoch)

        # Learning rate
        writer.add_scalar("Learning_Rate/LR", lr, epoch)

        # Box metrics
        writer.add_scalar("Metrics/Box_Precision", precision, epoch)
        writer.add_scalar("Metrics/Box_Recall", recall, epoch)
        writer.add_scalar("Metrics/Box_mAP@0.5", map_50, epoch)
        writer.add_scalar("Metrics/Box_mAP@0.5:0.95", map_50_95, epoch)

        # Mask metrics
        writer.add_scalar("Metrics/Mask_Precision", mask_precision, epoch)
        writer.add_scalar("Metrics/Mask_Recall", mask_recall, epoch)
        writer.add_scalar("Metrics/Mask_mAP@0.5", mask_map_50, epoch)
        writer.add_scalar("Metrics/Mask_mAP@0.5:0.95", mask_map_50_95, epoch)

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:2d}: Box Loss={box_loss:.4f}, Seg Loss={seg_loss:.4f}, "
                f"mAP@0.5={map_50:.3f}, Mask mAP@0.5={mask_map_50:.3f}"
            )

    writer.close()

    print("\nSynthetic TensorBoard logs created!")
    print(f"To view: tensorboard --logdir={log_dir}")
    print("Available metrics:")
    print("- Loss curves: Box regression, Segmentation, Objectness, Classification")
    print("- Learning rate schedule")
    print("- Accuracy metrics: Precision, Recall, mAP@0.5, mAP@0.5:0.95")
    print("- Both box detection and instance segmentation metrics")

    return log_dir


def create_precision_recall_plots(save_dir="runs/train-seg/food_segmentation"):
    """Create precision-recall curve plots."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic PR curves for different classes
    class_names = ["protein", "carbohydrate", "fruit", "dessert", "flatware", "vegetable", "sauce", "soup", "snack"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Box detection PR curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        # Generate realistic PR curve
        recall = np.linspace(0, 1, 101)
        # Higher performing classes have better curves
        performance_factor = 0.7 + 0.3 * np.random.random()
        precision = performance_factor * (1 - 0.3 * recall) * np.exp(-0.5 * recall)
        precision = np.clip(precision, 0, 1)

        # Calculate AP (area under curve)
        ap = np.trapz(precision, recall)

        ax1.plot(recall, precision, color=color, linewidth=2, label=f"{class_name} AP={ap:.3f}")

    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Curves - Box Detection")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Segmentation PR curves (slightly lower performance)
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        recall = np.linspace(0, 1, 101)
        performance_factor = 0.6 + 0.3 * np.random.random()
        precision = performance_factor * (1 - 0.4 * recall) * np.exp(-0.6 * recall)
        precision = np.clip(precision, 0, 1)

        ap = np.trapz(precision, recall)

        ax2.plot(recall, precision, color=color, linewidth=2, label=f"{class_name} AP={ap:.3f}")

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves - Instance Segmentation")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    # Save plot
    pr_path = save_dir / "precision_recall_curves.png"
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Precision-Recall curves saved to: {pr_path}")
    return pr_path


if __name__ == "__main__":
    # Create synthetic logs
    log_dir = create_synthetic_training_logs()

    # Create PR curves
    pr_path = create_precision_recall_plots()

    print("\nDemo metrics created successfully!")
    print(f"1. TensorBoard logs: {log_dir}")
    print(f"2. PR curves: {pr_path}")
    print(f"3. Start TensorBoard: tensorboard --logdir={log_dir}")
