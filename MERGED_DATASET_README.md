# Merged Food Dataset for YOLO Training

## Overview
This project merges bounding box and segmentation annotations for food classification and organizes them into a proper YOLO training format.

## Dataset Information
- **Total Images**: 120 (96 train, 24 validation)
- **Total Annotations**: 2,924 (2,430 train, 494 validation)
- **Classes**: 9 food categories
- **Annotation Types**: Both bounding boxes and segmentation polygons

### Classes
1. `protein` - 1,206 annotations
2. `carbohydrate` - 278 annotations  
3. `fruit` - 58 annotations
4. `dessert` - 42 annotations
5. `flatware` - 124 annotations
6. `vegetable` - 1,140 annotations
7. `sauce` - 40 annotations
8. `soup` - 18 annotations
9. `snack` - 18 annotations

## Directory Structure
```
yolo_dataset_final/
├── data.yaml                 # Dataset configuration
├── images/
│   ├── train/               # Training images (96 files)
│   └── val/                 # Validation images (24 files)
└── labels/
    ├── train/               # Training labels (96 files)
    └── val/                 # Validation labels (24 files)
```

## Files Created

### Core Scripts
1. **`merge_bbox_seg_annotations.py`** - Merges bounding box and segmentation annotations
2. **`organize_final_yolo_dataset.py`** - Organizes merged data into YOLO format
3. **`train_final_merged_model.py`** - Training script optimized for food classification
4. **`dataset_summary.py`** - Generates dataset statistics and summary

### Dataset Directories
- **`merged_dataset_final/`** - Raw merged annotations
- **`yolo_dataset_final/`** - Final YOLO-formatted dataset

## Usage Instructions

### 1. Quick Training (Recommended)
```bash
python train_final_merged_model.py
```
This will train YOLOv5s for 100 epochs with optimized hyperparameters.

### 2. Custom Training Options
```bash
# Train larger model
python train_final_merged_model.py --model l --epochs 150

# Custom batch size and epochs
python train_final_merged_model.py --epochs 200 --batch-size 32

# Resume training
python train_final_merged_model.py --resume

# Validate dataset only
python train_final_merged_model.py --validate-only
```

### 3. Manual Training (Using original train.py)
```bash
python train.py --data yolo_dataset_final/data.yaml --weights yolov5s.pt --epochs 100
```

## Training Configuration

### Model Sizes Available
- `n` - Nano (smallest, fastest)
- `s` - Small (default, good balance)
- `m` - Medium
- `l` - Large  
- `x` - Extra Large (best accuracy)

### Optimized Hyperparameters
The training script includes optimized hyperparameters for food classification:
- Learning rate: 0.01
- Momentum: 0.937
- Weight decay: 0.0005
- Data augmentation optimized for food images

## Dataset Statistics

### Annotation Distribution
- **Complex polygons**: 1,215 (detailed segmentation masks)
- **Simple polygons**: 1,215 (8-point bounding polygons)

### Class Balance
- **Most common**: Protein (996), Vegetable (940)
- **Least common**: Soup (18), Snack (16)

## Results Location
Training results will be saved to:
```
runs/train/final_merged_food_model/
├── weights/
│   ├── best.pt          # Best weights
│   └── last.pt          # Last weights
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
└── val_batch*.jpg       # Validation predictions
```

## Next Steps
1. **Start Training**: Run the training script
2. **Monitor Progress**: Check TensorBoard logs
3. **Evaluate Results**: Review validation metrics
4. **Deploy Model**: Use best.pt for inference

## Validation
Dataset validation confirms:
- ✅ All images have corresponding labels
- ✅ Proper YOLO format structure
- ✅ Balanced train/validation split
- ✅ All annotation files are valid

## Notes
- The merged dataset combines both bounding box and segmentation data
- Train/validation split is 80/20 with random seed 42 for reproducibility
- Images support both .jpg and .png formats
- All coordinates are normalized to [0,1] range as required by YOLO
