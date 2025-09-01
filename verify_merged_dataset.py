#!/usr/bin/env python3
"""Verification script for merged annotations dataset."""

import os
from collections import Counter, defaultdict
from pathlib import Path


def analyze_annotation_file(file_path):
    """Analyze a single annotation file."""
    annotations = []

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                annotations.append(
                    {
                        "class_id": class_id,
                        "coord_count": len(coords),
                        "format": "bbox" if len(coords) == 4 else "polygon",
                    }
                )

    return annotations


def analyze_dataset(dataset_path):
    """Analyze the entire dataset."""
    labels_dir = Path(dataset_path) / "labels" / "train"

    if not labels_dir.exists():
        print(f"Labels directory not found: {labels_dir}")
        return

    total_files = 0
    total_annotations = 0
    class_counts = Counter()
    format_counts = Counter()
    coord_stats = defaultdict(int)

    print(f"Analyzing dataset: {dataset_path}")
    print(f"Labels directory: {labels_dir}")
    print("-" * 50)

    for txt_file in sorted(labels_dir.glob("*.txt")):
        total_files += 1
        annotations = analyze_annotation_file(txt_file)
        total_annotations += len(annotations)

        for ann in annotations:
            class_counts[ann["class_id"]] += 1
            format_counts[ann["format"]] += 1
            coord_stats[ann["coord_count"]] += 1

    print("Dataset Statistics:")
    print(f"  Total files: {total_files}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Average annotations per file: {total_annotations / total_files:.2f}")
    print()

    print("Class Distribution:")
    class_names = {
        0: "protein",
        1: "carbohydrate",
        2: "fruit",
        3: "dessert",
        4: "flatware",
        5: "vegetable",
        6: "sauce",
        7: "soup",
        8: "snack",
    }
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_annotations) * 100
        class_name = class_names.get(class_id, f"unknown_{class_id}")
        print(f"  Class {class_id} ({class_name}): {count} ({percentage:.1f}%)")
    print()

    print("Format Distribution:")
    for format_type, count in format_counts.items():
        percentage = (count / total_annotations) * 100
        print(f"  {format_type}: {count} ({percentage:.1f}%)")
    print()

    print("Coordinate Count Distribution:")
    for coord_count in sorted(coord_stats.keys()):
        count = coord_stats[coord_count]
        percentage = (count / total_annotations) * 100
        if coord_count == 4:
            desc = " (bounding box format)"
        elif coord_count == 8:
            desc = " (simple rectangle polygon)"
        else:
            desc = f" (complex polygon with {coord_count // 2} points)"
        print(f"  {coord_count} coordinates{desc}: {count} ({percentage:.1f}%)")


def main():
    datasets = [
        "merged_dataset",
        "merged_dataset_combined",
        "dataset/annotation/boundingbox",
        "dataset/annotation/segmentation",
    ]

    for dataset in datasets:
        if os.path.exists(dataset):
            print("=" * 60)
            analyze_dataset(dataset)
            print()
        else:
            print(f"Dataset not found: {dataset}")


if __name__ == "__main__":
    main()
