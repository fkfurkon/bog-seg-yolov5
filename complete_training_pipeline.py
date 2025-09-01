#!/usr/bin/env python
"""
Complete YOLOv5 Segmentation Training Pipeline
This script provides a complete training pipeline with:
1. Model training with tensorboard logging
2. Validation and metrics computation
3. Instance segmentation inference
4. Visualization generation.
"""

import argparse
import sys
from pathlib import Path

# Add YOLOv5 paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.downloads import attempt_download
from utils.general import LOGGER, colorstr


def download_pretrained_weights(weights_path="yolov5s-seg.pt"):
    """Download pretrained weights if they don't exist."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        LOGGER.info(f"{colorstr('Downloading:')} {weights_path}...")
        attempt_download(weights_path, repo="ultralytics/yolov5")
    return weights_path


def run_training(epochs=100, batch_size=16, img_size=640, device="", name="food_segmentation"):
    """Run the training process."""
    LOGGER.info(f"{colorstr('Training:')} Starting YOLOv5 segmentation training...")

    # Download pretrained weights
    weights = download_pretrained_weights("yolov5s-seg.pt")

    # Set up training command
    cmd = [
        sys.executable,
        "segment/train.py",
        "--weights",
        str(weights),
        "--data",
        "food_dataset.yaml",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--imgsz",
        str(img_size),
        "--project",
        "runs/train-seg",
        "--name",
        name,
        "--device",
        device,
        "--cache",
        "ram",
        "--workers",
        "8",
        "--optimizer",
        "SGD",
        "--save-period",
        "10",  # Save every 10 epochs
        "--exist-ok",
        "--mask-ratio",
        "4",
        "--cos-lr",  # Use cosine learning rate
    ]

    # Run training
    import subprocess

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        LOGGER.error(f"{colorstr('Error:')} Training failed: {result.stderr}")
        raise RuntimeError("Training failed")

    LOGGER.info(f"{colorstr('Training:')} Training completed successfully!")

    # Find the trained model
    model_dir = Path("runs/train-seg") / name
    best_weights = model_dir / "weights" / "best.pt"

    if not best_weights.exists():
        raise FileNotFoundError(f"Trained model not found at {best_weights}")

    return best_weights, model_dir


def run_validation(weights, data="food_dataset.yaml", batch_size=16, img_size=640, device=""):
    """Run validation on the trained model."""
    LOGGER.info(f"{colorstr('Validation:')} Running validation...")

    cmd = [
        sys.executable,
        "segment/val.py",
        "--weights",
        str(weights),
        "--data",
        data,
        "--batch-size",
        str(batch_size),
        "--imgsz",
        str(img_size),
        "--device",
        device,
        "--save-txt",
        "--save-conf",
        "--plots",
        "--task",
        "val",
    ]

    import subprocess

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        LOGGER.error(f"{colorstr('Error:')} Validation failed: {result.stderr}")
        raise RuntimeError("Validation failed")

    LOGGER.info(f"{colorstr('Validation:')} Validation completed successfully!")
    return result.stdout


def create_sample_inference(weights, source_dir, output_dir, device=""):
    """Create sample inference results."""
    LOGGER.info(f"{colorstr('Inference:')} Creating sample inference results...")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference script
    cmd = [
        sys.executable,
        "instance_segmentation_inference.py",
        "--weights",
        str(weights),
        "--source",
        str(source_dir),
        "--data",
        "food_dataset.yaml",
        "--project",
        str(output_dir.parent),
        "--name",
        output_dir.name,
        "--device",
        device,
        "--save-txt",
        "--save-conf",
        "--conf-thres",
        "0.25",
        "--iou-thres",
        "0.45",
        "--line-thickness",
        "3",
        "--exist-ok",
    ]

    import subprocess

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        LOGGER.error(f"{colorstr('Error:')} Inference failed: {result.stderr}")
        raise RuntimeError("Inference failed")

    LOGGER.info(f"{colorstr('Inference:')} Inference completed successfully!")
    return output_dir


def setup_tensorboard_logging(log_dir):
    """Set up TensorBoard logging."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple script to start TensorBoard
    tb_script = log_dir / "start_tensorboard.sh"
    with open(tb_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"tensorboard --logdir={log_dir} --host=0.0.0.0 --port=6006\n")

    tb_script.chmod(0o755)

    LOGGER.info(f"{colorstr('TensorBoard:')} Log directory: {log_dir}")
    LOGGER.info(f"{colorstr('TensorBoard:')} Start with: bash {tb_script}")
    LOGGER.info(f"{colorstr('TensorBoard:')} Or run: tensorboard --logdir={log_dir}")

    return log_dir


def create_results_summary(model_dir, val_output, inference_dir):
    """Create a summary of training results."""
    summary_file = model_dir / "training_summary.md"

    with open(summary_file, "w") as f:
        f.write("# YOLOv5 Segmentation Training Results\n\n")

        f.write("## Training Configuration\n")
        f.write("- Model: YOLOv5s-seg\n")
        f.write("- Dataset: Food segmentation dataset\n")
        f.write("- Classes: 9 (protein, carbohydrate, fruit, dessert, flatware, vegetable, sauce, soup, snack)\n\n")

        f.write("## Files Generated\n")
        f.write(f"- Trained model: `{model_dir}/weights/best.pt`\n")
        f.write(f"- Training plots: `{model_dir}/`\n")
        f.write(f"- Validation results: `{model_dir}/val/`\n")
        f.write(f"- Inference examples: `{inference_dir}/`\n")
        f.write(f"- TensorBoard logs: `{model_dir}/tensorboard/`\n\n")

        f.write("## TensorBoard Metrics Available\n")
        f.write("- Box regression loss\n")
        f.write("- Objectness loss\n")
        f.write("- Classification loss\n")
        f.write("- Segmentation loss\n")
        f.write("- Learning rate\n")
        f.write("- Precision/Recall curves\n")
        f.write("- mAP@0.5 and mAP@0.5:0.95\n\n")

        f.write("## Usage\n")
        f.write("1. View training progress: `tensorboard --logdir=runs/train-seg/food_segmentation`\n")
        f.write(
            "2. Run inference: `python instance_segmentation_inference.py --weights runs/train-seg/food_segmentation/weights/best.pt --source path/to/images`\n"
        )
        f.write(
            "3. Validate model: `python segment/val.py --weights runs/train-seg/food_segmentation/weights/best.pt --data food_dataset.yaml`\n\n"
        )

    LOGGER.info(f"{colorstr('Summary:')} Results summary saved to {summary_file}")
    return summary_file


def main():
    parser = argparse.ArgumentParser(description="Complete YOLOv5 Segmentation Training Pipeline")
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--img-size", type=int, default=640, help="image size")
    parser.add_argument("--device", type=str, default="", help="device (cpu, 0, 1, etc.)")
    parser.add_argument("--name", type=str, default="food_segmentation", help="experiment name")
    parser.add_argument("--inference-samples", type=int, default=4, help="number of inference samples to create")
    parser.add_argument("--skip-training", action="store_true", help="skip training (use existing model)")
    parser.add_argument("--weights", type=str, default="", help="path to existing weights (if skip-training)")

    args = parser.parse_args()

    try:
        if args.skip_training and args.weights:
            # Use existing weights
            best_weights = Path(args.weights)
            model_dir = best_weights.parent.parent
            LOGGER.info(f"{colorstr('Using:')} Existing weights {best_weights}")
        else:
            # Run training
            best_weights, model_dir = run_training(
                epochs=args.epochs,
                batch_size=args.batch_size,
                img_size=args.img_size,
                device=args.device,
                name=args.name,
            )

        # Set up TensorBoard logging
        tb_log_dir = setup_tensorboard_logging(model_dir / "tensorboard")

        # Run validation
        val_output = run_validation(
            weights=best_weights, batch_size=args.batch_size, img_size=args.img_size, device=args.device
        )

        # Create inference samples
        inference_dir = create_sample_inference(
            weights=best_weights,
            source_dir="dataset/image",
            output_dir=f"runs/inference/{args.name}",
            device=args.device,
        )

        # Create results summary
        summary_file = create_results_summary(model_dir, val_output, inference_dir)

        LOGGER.info(f"{colorstr('Complete:')} Training pipeline completed successfully!")
        LOGGER.info(f"{colorstr('Model:')} Best weights saved at {best_weights}")
        LOGGER.info(f"{colorstr('Summary:')} Results summary at {summary_file}")
        LOGGER.info(f"{colorstr('TensorBoard:')} Start TensorBoard with: tensorboard --logdir={tb_log_dir}")
        LOGGER.info(f"{colorstr('Inference:')} Sample results at {inference_dir}")

    except Exception as e:
        LOGGER.error(f"{colorstr('Error:')} Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
