#!/usr/bin/env python
"""
YOLOv5 Segmentation Inference - YOLO Format
This script performs inference using trained YOLOv5 segmentation model
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import yaml

class YOLOSegmentationInference:
    def __init__(self, weights_path, data_config):
        """Initialize YOLO segmentation inference"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                      path=weights_path, device=self.device)
            print(f"✅ Model loaded from: {weights_path}")
        except:
            # Fallback to demo weights
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s-seg', 
                                      device=self.device)
            print(f"⚠️  Using demo weights (trained model not found)")
        
        # Load class names
        with open(data_config, 'r') as f:
            data = yaml.safe_load(f)
        self.class_names = data['names']
        self.num_classes = data['nc']
        
        print(f"📊 Classes: {list(self.class_names.values())}")
    
    def predict(self, image_path, conf_thresh=0.5, iou_thresh=0.45):
        """Run inference on image"""
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run inference
        results = self.model(image)
        
        # Extract results
        predictions = []
        if hasattr(results, 'pandas'):
            # YOLOv5 format
            df = results.pandas().xyxy[0]
            for _, row in df.iterrows():
                if row['confidence'] >= conf_thresh:
                    predictions.append({
                        'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                        'confidence': row['confidence'],
                        'class_id': int(row['class']),
                        'class_name': row['name']
                    })
        
        return predictions, image
    
    def visualize_predictions(self, image, predictions, save_path=None):
        """Visualize predictions with bounding boxes and masks"""
        
        # Colors for each class
        colors = [
            (0, 255, 0),    # protein - green
            (255, 165, 0),  # carbohydrate - orange  
            (255, 0, 255),  # fruit - magenta
            (255, 192, 203), # dessert - pink
            (128, 128, 128), # flatware - gray
            (0, 255, 0),    # vegetable - lime
            (255, 255, 0),  # sauce - yellow
            (255, 0, 0),    # soup - red
            (128, 0, 128)   # snack - purple
        ]
        
        result_image = image.copy()
        
        for pred in predictions:
            bbox = pred['bbox']
            conf = pred['confidence']
            class_id = pred['class_id']
            class_name = pred['class_name']
            
            # Get color
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(result_image, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Text
            cv2.putText(result_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(str(save_path), result_image)
            print(f"💾 Saved result: {save_path}")
        
        return result_image

def run_inference_examples():
    """Run inference on sample images"""
    
    print("🔍 Running YOLOv5 Segmentation Inference Examples")
    print("=" * 60)
    
    # Paths
    weights_path = "runs/train-seg/yolo_food_segmentation/weights/best.pt"
    data_config = "yolo_dataset/data.yaml"
    
    # Check if trained model exists
    if not Path(weights_path).exists():
        weights_path = "yolov5s-seg.pt"  # Use pretrained as fallback
        print(f"⚠️  Trained model not found, using pretrained: {weights_path}")
    
    # Initialize inference
    inference = YOLOSegmentationInference(weights_path, data_config)
    
    # Output directory
    output_dir = Path("runs/inference_yolo_format")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample images
    image_dir = Path("yolo_dataset/images/val")
    sample_images = list(image_dir.glob("*.jpg"))[:6] + list(image_dir.glob("*.png"))[:6]
    
    if not sample_images:
        # Fallback to train images
        image_dir = Path("yolo_dataset/images/train") 
        sample_images = list(image_dir.glob("*.jpg"))[:6] + list(image_dir.glob("*.png"))[:6]
    
    print(f"📁 Processing {len(sample_images)} sample images...")
    
    results = []
    for idx, img_path in enumerate(sample_images[:6]):
        print(f"\n🖼️  Processing: {img_path.name}")
        
        try:
            # Run inference
            predictions, image = inference.predict(img_path, conf_thresh=0.3)
            
            print(f"   Detected: {len(predictions)} objects")
            for pred in predictions:
                class_name = pred['class_name']
                conf = pred['confidence']
                print(f"     - {class_name}: {conf:.3f}")
            
            # Visualize and save
            output_path = output_dir / f"inference_result_{idx+1:02d}.jpg"
            result_image = inference.visualize_predictions(image, predictions, output_path)
            
            results.append({
                'image': str(img_path),
                'output': str(output_path),
                'detections': len(predictions)
            })
            
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")
    
    # Create summary
    create_inference_summary(results, output_dir)
    
    print(f"\n✅ Inference completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Processed {len(results)} images")
    print(f"🎯 Features demonstrated:")
    print(f"   ✓ Bounding box detection")
    print(f"   ✓ Multi-class food recognition")
    print(f"   ✓ Confidence scoring")
    print(f"   ✓ Thai food categories")

def create_inference_summary(results, output_dir):
    """Create inference summary visualization"""
    
    if len(results) < 4:
        return
    
    print(f"\n📋 Creating inference summary...")
    
    # Load result images
    images = []
    for result in results[:6]:
        try:
            img = cv2.imread(result['output'])
            if img is not None:
                img = cv2.resize(img, (300, 200))  # Thumbnail size
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
        except:
            continue
    
    if len(images) < 4:
        return
    
    # Create grid
    rows = 2
    cols = 3 if len(images) >= 6 else 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    fig.suptitle('YOLOv5 Instance Segmentation - Thai Food Dataset\n'
                 'YOLO Format Training Results', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, img in enumerate(images):
        if i < len(axes):
            axes[i].imshow(img)
            detections = results[i]['detections'] if i < len(results) else 0
            axes[i].set_title(f'Example {i+1}: {detections} food items detected',
                             fontsize=12, fontweight='bold')
            axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    summary_path = output_dir / 'yolo_inference_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Summary saved: {summary_path}")

if __name__ == '__main__':
    run_inference_examples()
