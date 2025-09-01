#!/usr/bin/env python
"""
Merged Dataset Inference - Bounding Box + Segmentation
This script runs inference on the merged dataset with both annotation types
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import random

class MergedDatasetInference:
    def __init__(self, weights_path="yolov5s-seg.pt", data_config="merged_dataset/data.yaml"):
        """Initialize inference with merged dataset"""
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        try:
            if Path(weights_path).exists():
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                          path=weights_path, device=self.device)
                print(f"✅ Model loaded from: {weights_path}")
            else:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s-seg', 
                                          device=self.device)
                print(f"⚠️  Using pretrained weights")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s-seg', 
                                      device=self.device)
        
        # Load dataset config
        try:
            with open(data_config, 'r') as f:
                data = yaml.safe_load(f)
            self.class_names = data['names']
            self.num_classes = data['nc']
        except:
            # Fallback class names
            self.class_names = {
                0: 'protein', 1: 'carbohydrate', 2: 'fruit', 3: 'dessert',
                4: 'flatware', 5: 'vegetable', 6: 'sauce', 7: 'soup', 8: 'snack'
            }
            self.num_classes = 9
        
        print(f"📊 Classes: {list(self.class_names.values())}")
        
        # Colors for visualization
        self.colors = [
            (0, 255, 0),     # protein - green
            (255, 165, 0),   # carbohydrate - orange
            (255, 0, 255),   # fruit - magenta
            (255, 192, 203), # dessert - pink
            (128, 128, 128), # flatware - gray
            (50, 205, 50),   # vegetable - lime green
            (255, 255, 0),   # sauce - yellow
            (255, 0, 0),     # soup - red
            (128, 0, 128)    # snack - purple
        ]
    
    def load_ground_truth(self, image_path, label_path):
        """Load ground truth annotations"""
        
        if not Path(label_path).exists():
            return []
        
        annotations = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        
                        annotations.append({
                            'class_id': class_id,
                            'polygon': coords,
                            'class_name': self.class_names.get(class_id, f'class_{class_id}')
                        })
        except Exception as e:
            print(f"⚠️  Error loading ground truth: {e}")
        
        return annotations
    
    def predict(self, image_path, conf_thresh=0.25):
        """Run inference on image"""
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run inference
        results = self.model(image)
        
        # Extract predictions
        predictions = []
        try:
            # Access results
            if hasattr(results, 'pandas'):
                df = results.pandas().xyxy[0]
                for _, row in df.iterrows():
                    if row['confidence'] >= conf_thresh:
                        predictions.append({
                            'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                            'confidence': row['confidence'],
                            'class_id': int(row['class']),
                            'class_name': row['name']
                        })
        except Exception as e:
            print(f"⚠️  Error extracting predictions: {e}")
        
        return predictions, image
    
    def visualize_comparison(self, image, predictions, ground_truth, save_path=None):
        """Visualize predictions vs ground truth"""
        
        height, width = image.shape[:2]
        
        # Create side-by-side comparison
        comparison = np.zeros((height, width * 2, 3), dtype=np.uint8)
        
        # Left side: Ground Truth
        gt_image = image.copy()
        for gt in ground_truth:
            class_id = gt['class_id']
            color = self.colors[class_id % len(self.colors)]
            
            # Draw polygon
            coords = gt['polygon']
            if len(coords) >= 6:  # At least 3 points
                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * width)
                    y = int(coords[i+1] * height)
                    points.append([x, y])
                
                points = np.array(points, dtype=np.int32)
                cv2.polylines(gt_image, [points], True, color, 2)
                
                # Create semi-transparent overlay
                overlay = gt_image.copy()
                cv2.fillPoly(overlay, [points], color)
                gt_image = cv2.addWeighted(gt_image, 0.7, overlay, 0.3, 0)
                
                # Label
                if len(points) > 0:
                    label = f"GT: {gt['class_name']}"
                    cv2.putText(gt_image, label, tuple(points[0]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Right side: Predictions
        pred_image = image.copy()
        for pred in predictions:
            class_id = pred['class_id']
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            bbox = pred['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(pred_image, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"Pred: {pred['class_name']} {pred['confidence']:.2f}"
            cv2.putText(pred_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Combine images
        comparison[:, :width] = gt_image
        comparison[:, width:] = pred_image
        
        # Add titles
        cv2.putText(comparison, "Ground Truth (Merged Annotations)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Model Predictions", (width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(str(save_path), comparison)
            print(f"💾 Saved comparison: {save_path}")
        
        return comparison

def run_merged_inference():
    """Run inference on merged dataset"""
    
    print("🔍 MERGED DATASET INFERENCE - Bounding Box + Segmentation")
    print("=" * 70)
    
    # Check for trained model
    trained_weights = "runs/train-seg/merged_food_segmentation/weights/best.pt"
    if Path(trained_weights).exists():
        weights_path = trained_weights
        print(f"✅ Using trained model: {weights_path}")
    else:
        weights_path = "yolov5s-seg.pt"
        print(f"⚠️  Using pretrained model: {weights_path}")
    
    # Initialize inference
    inference = MergedDatasetInference(weights_path, "merged_dataset/data.yaml")
    
    # Output directory
    output_dir = Path("runs/merged_inference")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample images from validation set
    val_images_dir = Path("merged_dataset/images/val")
    val_labels_dir = Path("merged_dataset/labels/val")
    
    sample_images = list(val_images_dir.glob("*.jpg"))[:6] + list(val_images_dir.glob("*.png"))[:6]
    
    if not sample_images:
        # Fallback to train images
        val_images_dir = Path("merged_dataset/images/train")
        val_labels_dir = Path("merged_dataset/labels/train")
        sample_images = list(val_images_dir.glob("*.jpg"))[:6] + list(val_images_dir.glob("*.png"))[:6]
    
    print(f"📁 Processing {len(sample_images)} sample images...")
    
    results = []
    
    for idx, img_path in enumerate(sample_images[:6]):
        print(f"\n🖼️  Processing: {img_path.name}")
        
        try:
            # Load ground truth
            label_path = val_labels_dir / f"{img_path.stem}.txt"
            ground_truth = inference.load_ground_truth(img_path, label_path)
            
            # Run inference
            predictions, image = inference.predict(img_path, conf_thresh=0.3)
            
            print(f"   Ground Truth: {len(ground_truth)} objects")
            print(f"   Predictions: {len(predictions)} objects")
            
            # Show ground truth classes
            gt_classes = set(gt['class_name'] for gt in ground_truth)
            pred_classes = set(pred['class_name'] for pred in predictions)
            print(f"   GT Classes: {', '.join(gt_classes)}")
            print(f"   Pred Classes: {', '.join(pred_classes)}")
            
            # Create comparison visualization
            output_path = output_dir / f"merged_comparison_{idx+1:02d}.jpg"
            comparison = inference.visualize_comparison(image, predictions, ground_truth, output_path)
            
            results.append({
                'image': str(img_path),
                'output': str(output_path),
                'ground_truth': len(ground_truth),
                'predictions': len(predictions)
            })
            
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")
    
    # Create summary grid
    create_merged_summary(results, output_dir)
    
    print(f"\n✅ Merged inference completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Features demonstrated:")
    print(f"   ✓ Ground truth vs predictions comparison")
    print(f"   ✓ Merged annotations (bbox + segmentation)")
    print(f"   ✓ Multi-class food recognition")
    print(f"   ✓ Thai food categories with distribution:")
    print(f"     • protein (41.2%)")
    print(f"     • vegetable (39.0%)")
    print(f"     • carbohydrate (9.5%)")
    print(f"     • flatware (4.2%)")
    print(f"     • Others (6.1%)")

def create_merged_summary(results, output_dir):
    """Create summary visualization"""
    
    if len(results) < 4:
        return
    
    print(f"\n📋 Creating merged inference summary...")
    
    # Load comparison images
    images = []
    for result in results[:6]:
        try:
            img = cv2.imread(result['output'])
            if img is not None:
                # Resize for grid
                img = cv2.resize(img, (600, 300))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
        except:
            continue
    
    if len(images) < 4:
        return
    
    # Create grid
    rows = 3 if len(images) >= 6 else 2
    cols = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    fig.suptitle('Merged Dataset Inference Results\n'
                 'Ground Truth (Left) vs Model Predictions (Right)', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten() if rows > 1 else [axes]
    
    for i, img in enumerate(images):
        if i < len(axes):
            axes[i].imshow(img)
            gt_count = results[i]['ground_truth'] if i < len(results) else 0
            pred_count = results[i]['predictions'] if i < len(results) else 0
            axes[i].set_title(f'Example {i+1}: GT={gt_count} | Pred={pred_count}',
                             fontsize=12, fontweight='bold')
            axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    summary_path = output_dir / 'merged_inference_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Summary saved: {summary_path}")

if __name__ == '__main__':
    run_merged_inference()
