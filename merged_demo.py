#!/usr/bin/env python
"""
Simple Merged Dataset Visualization
Creates demo visualizations showing the merged annotation capabilities
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import yaml

def create_merged_demo_visualization():
    """Create demo visualization showing merged dataset capabilities"""
    
    print("🎨 Creating Merged Dataset Demo Visualization")
    print("=" * 50)
    
    # Load dataset info
    with open('merged_dataset/data.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    class_names = data['names']
    class_colors = [
        (0, 255, 0),     # protein - green
        (255, 165, 0),   # carbohydrate - orange
        (255, 0, 255),   # fruit - magenta
        (255, 192, 203), # dessert - pink
        (128, 128, 128), # flatware - gray
        (0, 255, 0),     # vegetable - lime
        (255, 255, 0),   # sauce - yellow
        (255, 0, 0),     # soup - red
        (128, 0, 128)    # snack - purple
    ]
    
    # Thai translations
    thai_names = {
        0: 'โปรตีน',
        1: 'คาร์โบไฮเดรต',
        2: 'ผลไม้',
        3: 'ของหวาน',
        4: 'ช้อนส้อม',
        5: 'ผัก',
        6: 'ซอส',
        7: 'ซุป',
        8: 'ขนม'
    }
    
    # Get sample images
    train_images = list(Path('merged_dataset/images/train').glob('*.jpg')) + \
                   list(Path('merged_dataset/images/train').glob('*.png'))
    
    sample_images = random.sample(train_images, min(6, len(train_images)))
    
    # Output directory
    output_dir = Path('runs/merged_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📸 Processing {len(sample_images)} images...")
    
    results = []
    for idx, img_path in enumerate(sample_images):
        print(f"   Processing: {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        height, width = image.shape[:2]
        
        # Resize for demo
        max_size = 640
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            height, width = new_height, new_width
        
        result_image = image.copy()
        
        # Load corresponding label file
        label_path = Path('merged_dataset/labels/train') / f"{img_path.stem}.txt"
        
        detections = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7:  # class_id + at least 6 coordinates
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        
                        # Convert normalized coordinates to pixel coordinates
                        pixel_coords = []
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                x = int(coords[i] * width)
                                y = int(coords[i + 1] * height)
                                pixel_coords.extend([x, y])
                        
                        if len(pixel_coords) >= 6:
                            detections.append({
                                'class_id': class_id,
                                'coords': pixel_coords
                            })
        
        # Draw annotations
        for detection in detections:
            class_id = detection['class_id']
            coords = detection['coords']
            
            if class_id >= len(class_colors):
                continue
                
            color = class_colors[class_id]
            class_name = class_names.get(class_id, f'class_{class_id}')
            thai_name = thai_names.get(class_id, '')
            
            # Draw polygon
            if len(coords) >= 6:
                points = np.array(coords).reshape(-1, 2).astype(np.int32)
                
                # Fill polygon with transparency
                overlay = result_image.copy()
                cv2.fillPoly(overlay, [points], color)
                cv2.addWeighted(overlay, 0.3, result_image, 0.7, 0, result_image)
                
                # Draw polygon outline
                cv2.polylines(result_image, [points], True, color, 2)
                
                # Calculate bounding box for label
                x_coords = coords[0::2]
                y_coords = coords[1::2]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Draw bounding box
                cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Draw label
                label = f"{thai_name} ({class_name})"
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Background for text
                cv2.rectangle(result_image,
                             (x_min, y_min - text_height - baseline - 5),
                             (x_min + text_width, y_min),
                             color, -1)
                
                # Text
                cv2.putText(result_image, label, (x_min, y_min - baseline - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Save result
        output_path = output_dir / f'merged_demo_{idx+1:02d}.jpg'
        cv2.imwrite(str(output_path), result_image)
        
        results.append({
            'image': str(img_path),
            'output': str(output_path),
            'detections': len(detections)
        })
        
        print(f"     → Saved: {output_path.name} ({len(detections)} objects)")
    
    # Create summary grid
    create_demo_summary_grid(results, output_dir)
    
    print(f"\n✅ Merged demo visualization completed!")
    print(f"📁 Results: {output_dir}")
    print(f"📊 Features shown:")
    print(f"   ✓ Merged bounding box + segmentation annotations")
    print(f"   ✓ Polygon segmentation masks")
    print(f"   ✓ Bounding box overlays")
    print(f"   ✓ Thai and English labels")
    print(f"   ✓ 9 food classes")
    print(f"   ✓ {len(results)} example images")

def create_demo_summary_grid(results, output_dir):
    """Create summary grid of demo results"""
    
    if len(results) < 4:
        return
    
    print(f"\n📋 Creating demo summary grid...")
    
    # Load images
    images = []
    for result in results[:6]:
        try:
            img = cv2.imread(result['output'])
            if img is not None:
                img = cv2.resize(img, (320, 240))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
        except:
            continue
    
    if len(images) < 4:
        return
    
    # Create grid
    rows = 2
    cols = 3 if len(images) >= 6 else 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    fig.suptitle('Merged Dataset Visualization - Bounding Box + Segmentation\n'
                 'Thai Food Instance Segmentation (9 Classes)\n'
                 'Combined Annotation Training Format', 
                 fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, img in enumerate(images):
        if i < len(axes):
            axes[i].imshow(img)
            detections = results[i]['detections'] if i < len(results) else 0
            axes[i].set_title(f'Example {i+1}: {detections} food items\nBbox + Polygon Annotations',
                             fontsize=11, fontweight='bold')
            axes[i].axis('off')
    
    # Hide extra subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save grid
    summary_path = output_dir / 'merged_demo_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Summary grid: {summary_path}")

def show_annotation_stats():
    """Show statistics about the merged annotations"""
    
    print(f"\n📊 MERGED DATASET STATISTICS")
    print("-" * 40)
    
    # Analyze training labels
    labels_dir = Path('merged_dataset/labels/train')
    label_files = list(labels_dir.glob('*.txt'))
    
    bbox_annotations = 0
    polygon_annotations = 0
    total_objects = 0
    class_counts = {}
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        class_id = int(parts[0])
                        coords = parts[1:]
                        
                        total_objects += 1
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        
                        # Determine annotation type by coordinate count
                        if len(coords) == 8:  # Rectangle (4 points)
                            bbox_annotations += 1
                        elif len(coords) > 8:  # Polygon (>4 points)
                            polygon_annotations += 1
        except:
            continue
    
    print(f"📁 Files: {len(label_files)}")
    print(f"🎯 Total objects: {total_objects}")
    print(f"📦 Bounding box annotations: {bbox_annotations}")
    print(f"🔺 Polygon annotations: {polygon_annotations}")
    
    print(f"\n📈 Class distribution:")
    class_names = ['protein', 'carbohydrate', 'fruit', 'dessert', 'flatware',
                   'vegetable', 'sauce', 'soup', 'snack']
    thai_names = ['โปรตีน', 'คาร์โบไฮเดรต', 'ผลไม้', 'ของหวาน', 'ช้อนส้อม',
                  'ผัก', 'ซอส', 'ซุป', 'ขนม']
    
    for class_id, count in sorted(class_counts.items()):
        if class_id < len(class_names):
            eng_name = class_names[class_id]
            thai_name = thai_names[class_id]
            percentage = (count / total_objects) * 100
            print(f"   {class_id}: {eng_name} ({thai_name}) - {count} objects ({percentage:.1f}%)")

if __name__ == '__main__':
    # Show statistics
    show_annotation_stats()
    
    # Create demo visualization
    create_merged_demo_visualization()
    
    print(f"\n🎯 MERGED DATASET READY!")
    print(f"✅ Annotations successfully merged")
    print(f"✅ Demo visualizations created")
    print(f"✅ Training in progress")
    print(f"📈 Monitor training: tensorboard --logdir runs/train-seg")
