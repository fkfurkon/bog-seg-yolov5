#!/usr/bin/env python
"""
Merge Bounding Box and Segmentation Annotations
This script merges bounding box and polygon segmentation annotations into unified YOLO format.
"""

import shutil
from pathlib import Path

import yaml


def parse_bbox_annotation(line):
    """Parse bounding box annotation line."""
    parts = line.strip().split()
    if len(parts) < 9:  # class_id + 8 coordinates (4 points)
        return None

    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:9]]  # Take first 8 coordinates (4 points)

    # Convert 4-point bbox to standard bbox format (x_center, y_center, width, height)
    x_coords = coords[0::2]  # x coordinates
    y_coords = coords[1::2]  # y coordinates

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # YOLO format: center_x, center_y, width, height
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return {"class_id": class_id, "bbox": [center_x, center_y, width, height], "type": "bbox"}


def parse_segmentation_annotation(line):
    """Parse segmentation annotation line."""
    parts = line.strip().split()
    if len(parts) < 7:  # class_id + at least 6 coordinates (3 points minimum)
        return None

    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]

    # Ensure even number of coordinates (pairs of x,y)
    if len(coords) % 2 != 0:
        coords = coords[:-1]  # Remove last coordinate if odd

    return {"class_id": class_id, "polygon": coords, "type": "segmentation"}


def merge_annotations(bbox_dir, seg_dir, output_dir):
    """Merge bounding box and segmentation annotations."""
    print("🔗 Merging Bounding Box and Segmentation Annotations")
    print("=" * 60)

    bbox_labels_dir = Path(bbox_dir) / "labels" / "train"
    seg_labels_dir = Path(seg_dir) / "labels" / "train"
    output_labels_dir = Path(output_dir) / "labels" / "train"

    # Create output directory
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Get all label files
    bbox_files = set(f.name for f in bbox_labels_dir.glob("*.txt"))
    seg_files = set(f.name for f in seg_labels_dir.glob("*.txt"))

    all_files = bbox_files.union(seg_files)

    print("📁 Files found:")
    print(f"   Bounding box files: {len(bbox_files)}")
    print(f"   Segmentation files: {len(seg_files)}")
    print(f"   Total unique files: {len(all_files)}")

    merged_count = 0
    bbox_only_count = 0
    seg_only_count = 0

    for filename in sorted(all_files):
        bbox_file = bbox_labels_dir / filename
        seg_file = seg_labels_dir / filename
        output_file = output_labels_dir / filename

        merged_annotations = []

        # Process bounding box annotations
        if bbox_file.exists():
            try:
                with open(bbox_file) as f:
                    for line in f:
                        if line.strip():
                            bbox_ann = parse_bbox_annotation(line)
                            if bbox_ann:
                                merged_annotations.append(bbox_ann)
            except Exception as e:
                print(f"⚠️  Error reading {bbox_file}: {e}")

        # Process segmentation annotations
        seg_annotations = []
        if seg_file.exists():
            try:
                with open(seg_file) as f:
                    for line in f:
                        if line.strip():
                            seg_ann = parse_segmentation_annotation(line)
                            if seg_ann:
                                seg_annotations.append(seg_ann)
            except Exception as e:
                print(f"⚠️  Error reading {seg_file}: {e}")

        # Merge strategy: Use segmentation if available, otherwise use bounding box
        final_annotations = []

        if seg_annotations:
            # Use segmentation annotations (they include implicit bounding boxes)
            final_annotations = seg_annotations
            if bbox_file.exists() and seg_file.exists():
                merged_count += 1
            elif seg_file.exists():
                seg_only_count += 1
        elif merged_annotations:
            # Use bounding box annotations only
            final_annotations = merged_annotations
            bbox_only_count += 1

        # Write merged annotations
        if final_annotations:
            with open(output_file, "w") as f:
                for ann in final_annotations:
                    if ann["type"] == "segmentation":
                        # Write segmentation format
                        coords_str = " ".join([f"{coord:.6f}" for coord in ann["polygon"]])
                        f.write(f"{ann['class_id']} {coords_str}\n")
                    else:
                        # Write bbox format (convert to segmentation-compatible format)
                        x_center, y_center, width, height = ann["bbox"]
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        # Create rectangle as polygon
                        polygon = [x1, y1, x2, y1, x2, y2, x1, y2]
                        coords_str = " ".join([f"{coord:.6f}" for coord in polygon])
                        f.write(f"{ann['class_id']} {coords_str}\n")

    print("\n📊 Merge Results:")
    print(f"   Files with both bbox & segmentation: {merged_count}")
    print(f"   Files with segmentation only: {seg_only_count}")
    print(f"   Files with bounding box only: {bbox_only_count}")
    print(f"   Total output files: {len(list(output_labels_dir.glob('*.txt')))}")

    return output_labels_dir


def create_merged_dataset():
    """Create complete merged dataset with images and labels."""
    print("\n📦 Creating Complete Merged Dataset")
    print("-" * 40)

    # Paths
    bbox_dir = Path("dataset/annotation/boundingbox")
    seg_dir = Path("dataset/annotation/segmentation")
    images_dir = Path("dataset/image")
    output_dir = Path("merged_dataset")

    # Create output structure
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Merge annotations
    merge_annotations(bbox_dir, seg_dir, output_dir)

    # Copy images
    print("\n📸 Copying Images...")
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    # Split images (80% train, 20% val)
    train_split = 0.8
    train_count = int(len(image_files) * train_split)

    train_images = image_files[:train_count]
    val_images = image_files[train_count:]

    # Copy train images
    train_list = []
    for img_path in train_images:
        dest_path = output_dir / "images" / "train" / img_path.name
        shutil.copy2(img_path, dest_path)
        train_list.append(f"./images/train/{img_path.name}")

    # Copy val images
    val_list = []
    for img_path in val_images:
        dest_path = output_dir / "images" / "val" / img_path.name
        shutil.copy2(img_path, dest_path)
        val_list.append(f"./images/val/{img_path.name}")

    # Create file lists
    with open(output_dir / "train.txt", "w") as f:
        f.write("\n".join(train_list))

    with open(output_dir / "val.txt", "w") as f:
        f.write("\n".join(val_list))

    # Create dataset config
    dataset_config = {
        "path": str(output_dir.absolute()),
        "train": "train.txt",
        "val": "val.txt",
        "nc": 9,
        "names": {
            0: "protein",
            1: "carbohydrate",
            2: "fruit",
            3: "dessert",
            4: "flatware",
            5: "vegetable",
            6: "sauce",
            7: "soup",
            8: "snack",
        },
    }

    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"✅ Merged dataset created at: {output_dir}")
    print("📁 Structure:")
    print("   ├── images/")
    print(f"   │   ├── train/ ({len(train_images)} images)")
    print(f"   │   └── val/ ({len(val_images)} images)")
    print("   ├── labels/")
    print("   │   ├── train/ (merged annotations)")
    print("   │   └── val/ (merged annotations)")
    print("   ├── data.yaml")
    print("   ├── train.txt")
    print("   └── val.txt")

    return output_dir


def analyze_annotations(directory):
    """Analyze annotation types and quality."""
    print(f"\n🔍 Analyzing Annotations in: {directory}")
    print("-" * 40)

    labels_dir = Path(directory) / "labels" / "train"

    if not labels_dir.exists():
        print(f"❌ Directory not found: {labels_dir}")
        return

    label_files = list(labels_dir.glob("*.txt"))

    bbox_count = 0
    seg_count = 0
    total_objects = 0
    class_distribution = {}

    for label_file in label_files:
        try:
            with open(label_file) as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            class_id = int(parts[0])
                            coords = parts[1:]

                            # Count class distribution
                            class_distribution[class_id] = class_distribution.get(class_id, 0) + 1
                            total_objects += 1

                            # Determine annotation type
                            if len(coords) == 8:  # Rectangle (4 points)
                                bbox_count += 1
                            elif len(coords) > 8:  # Polygon (>4 points)
                                seg_count += 1
        except Exception as e:
            print(f"⚠️  Error reading {label_file}: {e}")

    print("📊 Annotation Analysis:")
    print(f"   Total files: {len(label_files)}")
    print(f"   Total objects: {total_objects}")
    print(f"   Bounding boxes: {bbox_count}")
    print(f"   Segmentation polygons: {seg_count}")

    print("\n📈 Class Distribution:")
    class_names = ["protein", "carbohydrate", "fruit", "dessert", "flatware", "vegetable", "sauce", "soup", "snack"]

    for class_id, count in sorted(class_distribution.items()):
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        percentage = (count / total_objects) * 100 if total_objects > 0 else 0
        print(f"   {class_id} ({class_name}): {count} objects ({percentage:.1f}%)")


if __name__ == "__main__":
    print("🔗 ANNOTATION MERGER - Bounding Box + Segmentation")
    print("=" * 60)

    # Create merged dataset
    merged_dir = create_merged_dataset()

    # Analyze the merged annotations
    analyze_annotations(merged_dir)

    print("\n✅ Merge Complete!")
    print(f"📁 Merged dataset: {merged_dir}")
    print(f"🚀 Ready for training with: {merged_dir}/data.yaml")
    print("💡 Use this for training:")
    print(
        f"   python segment/train.py --data {merged_dir}/data.yaml --cfg models/segment/yolov5s-seg.yaml --weights yolov5s-seg.pt"
    )
