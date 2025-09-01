#!/usr/bin/env python
"""
YOLOv5 Instance Segmentation Inference with Visualization
This script performs inference with trained YOLOv5 segmentation model and creates
visualizations showing both bounding boxes and segmentation masks overlaid on images.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import torch
import random

# Add YOLOv5 paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, 
                          colorstr, cv2, increment_path, non_max_suppression, scale_boxes, 
                          scale_segments, strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode

class InstanceSegmentationVisualizer:
    def __init__(self, weights, data, device='', imgsz=640, conf_thres=0.25, iou_thres=0.45):
        """Initialize the visualizer with model weights and parameters"""
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=data, fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Generate colors for classes
        self.colors = [colors(i, True) for i in range(len(self.names))]
        
        # Warm up
        if self.pt:
            self.model.warmup(imgsz=(1, 3, *self.imgsz))
    
    def create_overlay_visualization(self, im0, boxes, masks, classes, confidences, save_path):
        """Create overlay visualization with bounding boxes and segmentation masks"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Instance Segmentation Results', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Image with bounding boxes only
        img_bbox = im0.copy()
        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            x1, y1, x2, y2 = box.astype(int)
            color = self.colors[int(cls)]
            cv2.rectangle(img_bbox, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f'{self.names[int(cls)]} {conf:.2f}'
            labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img_bbox, (x1, y1-labelSize[1]-10), (x1+labelSize[0], y1), color, -1)
            cv2.putText(img_bbox, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        axes[0, 1].imshow(cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Bounding Boxes Only')
        axes[0, 1].axis('off')
        
        # Image with segmentation masks only
        img_mask = im0.copy()
        mask_overlay = np.zeros_like(img_mask, dtype=np.uint8)
        
        for i, (mask, cls) in enumerate(zip(masks, classes)):
            color = self.colors[int(cls)]
            # Create colored mask
            colored_mask = np.zeros_like(img_mask, dtype=np.uint8)
            colored_mask[mask > 0] = color
            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 0.6, 0)
        
        img_mask_only = cv2.addWeighted(img_mask, 0.4, mask_overlay, 0.6, 0)
        axes[1, 0].imshow(cv2.cvtColor(img_mask_only, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Segmentation Masks Only')
        axes[1, 0].axis('off')
        
        # Combined: bounding boxes + segmentation masks
        img_combined = cv2.addWeighted(img_bbox, 0.7, mask_overlay, 0.3, 0)
        
        axes[1, 1].imshow(cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Combined: Boxes + Masks')
        axes[1, 1].axis('off')
        
        # Add legend
        legend_elements = []
        for i, name in enumerate(self.names.values()):
            color = [c/255.0 for c in self.colors[i]]  # Normalize for matplotlib
            legend_elements.append(patches.Patch(color=color, label=name))
        
        axes[1, 1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return img_combined
    
    @smart_inference_mode()
    def run_inference(self, source, save_dir='runs/inference', save_txt=False, save_conf=False, 
                     line_thickness=3, hide_labels=False, hide_conf=False, retina_masks=False):
        """Run inference on source images/directory"""
        
        save_dir = increment_path(Path(save_dir), exist_ok=True)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
        # Load dataset
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        
        # Initialize results storage
        results = []
        
        dt = (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            
            # Inference
            with dt[1]:
                pred, proto = self.model(im, augment=False, visualize=False)[:2]
            
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, 
                                         max_det=1000, nm=32)
            
            # Process predictions
            for i, det in enumerate(pred):  # per image
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                
                annotator = Annotator(im0, line_width=line_thickness, example=str(self.names))
                
                if len(det):
                    if retina_masks:
                        # scale bbox first then crop masks
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                        masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                    else:
                        masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    
                    # Segments
                    segments = [scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                               for x in reversed(masks2segments(masks))]
                    
                    # Collect data for visualization
                    boxes = det[:, :4].cpu().numpy()
                    classes = det[:, 5].cpu().numpy()
                    confidences = det[:, 4].cpu().numpy()
                    masks_np = masks.cpu().numpy()
                    
                    # Store results
                    result = {
                        'path': str(p),
                        'boxes': boxes,
                        'masks': masks_np,
                        'classes': classes,
                        'confidences': confidences,
                        'segments': segments
                    }
                    results.append(result)
                    
                    # Create overlay visualization
                    viz_path = str(save_dir / f'{p.stem}_visualization.png')
                    combined_img = self.create_overlay_visualization(
                        im0, boxes, masks_np, classes, confidences, viz_path
                    )
                    
                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    # Write results
                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        if save_txt:  # Write to file
                            seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                            line = (cls, *seg, conf) if save_conf else (cls, *seg)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    # Mask plotting
                    annotator.masks(
                        masks,
                        colors=[colors(x, True) for x in det[:, 5]],
                        im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(self.device).permute(2, 0, 1).flip(0).contiguous() / 255
                        if retina_masks else im[i]
                    )
                    
                    # Write bounding boxes
                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        c = int(cls)  # integer class
                        label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                
                # Save results
                im0 = annotator.result()
                cv2.imwrite(save_path, im0)
                
            # Print time (inference-only)
            LOGGER.info(f'{s}{dt[1].dt * 1E3:.1f}ms')
        
        # Print results
        t = tuple(x.t / len(dataset) * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
        
        return results, save_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default='food_dataset.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    
    # Initialize visualizer
    visualizer = InstanceSegmentationVisualizer(
        weights=opt.weights[0],
        data=opt.data,
        device=opt.device,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres
    )
    
    # Run inference
    results, save_dir = visualizer.run_inference(
        source=opt.source,
        save_dir=f"{opt.project}/{opt.name}",
        save_txt=opt.save_txt,
        save_conf=opt.save_conf,
        line_thickness=opt.line_thickness,
        hide_labels=opt.hide_labels,
        hide_conf=opt.hide_conf,
        retina_masks=opt.retina_masks
    )
    
    LOGGER.info(f'Instance segmentation completed. Results saved to {save_dir}')
    LOGGER.info(f'Processed {len(results)} images with detections')

if __name__ == '__main__':
    main()
