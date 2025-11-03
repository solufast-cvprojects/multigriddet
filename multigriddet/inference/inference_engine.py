#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference engine for MultiGridDet models.
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from ..config import ConfigLoader, build_model_for_inference
from ..utils.anchors import load_anchors, load_classes
from ..utils.tf_optimization import optimize_tf_gpu
from ..postprocess.multigrid_decode import MultiGridDecoder
from ..utils.preprocessing import preprocess_image
from ..utils.visualization import get_colors, draw_boxes


class MultiGridInference:
    """Inference engine for MultiGridDet models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize inference engine with configuration.
        
        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.model = None
        self.class_names = None
        self.anchors = None
        self.colors = None
        
        # Initialize TensorFlow optimizations
        optimize_tf_gpu()
        
        # Load model configuration
        model_config_path = config['model_config']
        self.model_config = ConfigLoader.load_config(model_config_path)
        
        # Merge model config with main config
        self.full_config = ConfigLoader.merge_configs(self.model_config, config)
        
        print("=" * 80)
        print("MultiGridDet Inference Engine Initialized")
        print("=" * 80)
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load the model and weights."""
        print("\n🔨 Loading model...")
        
        # Get weights path
        weights_path = self.config.get('weights_path')
        if not weights_path:
            raise ValueError("weights_path not specified in config")
        
        # Load classes and anchors
        classes_path = self.full_config['model']['preset'].get('classes_path')
        if not classes_path:
            # Try to get from data config
            classes_path = self.full_config.get('data', {}).get('classes_path')
        
        if not classes_path:
            raise ValueError("classes_path not found in config")
        
        anchors_path = self.model_config['model']['preset']['anchors_path']
        
        self.class_names = load_classes(classes_path)
        self.anchors = load_anchors(anchors_path)
        self.colors = get_colors(len(self.class_names))
        
        # Initialize decoder
        input_shape = tuple(self.model_config['model']['preset'].get('input_shape', [608, 608, 3])[:2])
        self.decoder = MultiGridDecoder(
            anchors=self.anchors,
            num_classes=len(self.class_names),
            input_shape=input_shape,
            rescore_confidence=True
        )
        
        print(f"   Classes: {len(self.class_names)}")
        print(f"   Anchors: {len(self.anchors)} scales")
        
        # Build and load model
        self.model = build_model_for_inference(self.full_config, weights_path)
        
        print(f"✓ Model loaded successfully\n")
        
    def predict_image(self, image_path: str) -> Tuple[np.ndarray, List, List, List]:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (annotated_image, boxes, classes, scores)
        """
        # Load image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get input shape
        input_shape = tuple(self.model_config['model']['preset']['input_shape'][:2])
        
        # Preprocess
        image_data = preprocess_image(image, input_shape)
        image_shape = tuple(reversed(image.size))
        
        # Run inference
        predictions = self.model.predict(image_data, verbose=0)
        
        # Post-process
        detection_config = self.config.get('detection', {})
        nms_method = detection_config.get('nms_method', 'diou')
        use_wbf = detection_config.get('use_wbf', False)
        boxes, classes, scores = self.decoder.postprocess(
            predictions,
            image_shape,
            input_shape,
            max_boxes=detection_config.get('max_boxes', 100),
            confidence=detection_config.get('confidence_threshold', 0.5),
            nms_threshold=detection_config.get('nms_threshold', 0.45),
            nms_method=nms_method,
            use_wbf=use_wbf,
            use_iol=detection_config.get('use_iol', True),
            return_xyxy=True
        )
        
        # Draw boxes
        image_array = np.array(image, dtype='uint8')
        annotated_image = draw_boxes(
            image_array,
            boxes,
            classes,
            scores,
            self.class_names,
            self.colors
        )
        
        return annotated_image, boxes, classes, scores
    
    def predict_video(self, video_path: str, output_path: str = None):
        """
        Run inference on a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        if output_path:
            video_config = self.config.get('video', {})
            fourcc_str = video_config.get('fourcc', 'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Get input shape
        input_shape = tuple(self.model_config['model']['preset']['input_shape'][:2])
        detection_config = self.config.get('detection', {})
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                # Preprocess
                image_data = preprocess_image(image, input_shape)
                image_shape = (height, width)
                
                # Run inference
                predictions = self.model.predict(image_data, verbose=0)
                
                # Post-process
                nms_method = detection_config.get('nms_method', 'diou')
                use_wbf = detection_config.get('use_wbf', False)
                boxes, classes, scores = self.decoder.postprocess(
                    predictions,
                    image_shape,
                    input_shape,
                    max_boxes=detection_config.get('max_boxes', 100),
                    confidence=detection_config.get('confidence_threshold', 0.5),
                    nms_threshold=detection_config.get('nms_threshold', 0.45),
                    nms_method=nms_method,
                    use_wbf=use_wbf,
                    use_iol=detection_config.get('use_iol', True),
                    return_xyxy=True
                )
                
                # Draw boxes
                frame_rgb = np.array(image, dtype='uint8')
                annotated_frame = draw_boxes(
                    frame_rgb,
                    boxes,
                    classes,
                    scores,
                    self.class_names,
                    self.colors
                )
                
                # Convert back to BGR
                annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                # Write frame
                if output_path:
                    out.write(annotated_frame_bgr)
                
                # Print progress
                if frame_count % 30 == 0:
                    print(f"   Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)", end='\r')
        
        finally:
            cap.release()
            if output_path:
                out.release()
            print(f"\n✓ Processed {frame_count} frames")
    
    def predict_camera(self, camera_id: int = 0, output_path: str = None):
        """
        Run inference on camera stream.
        
        Args:
            camera_id: Camera device ID
            output_path: Path to save output video
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")
        
        # Get camera properties
        camera_config = self.config.get('camera', {})
        resolution = camera_config.get('resolution', [1280, 720])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📷 Camera: {width}x{height}")
        print("Press 'q' to quit")
        
        # Setup video writer
        if output_path:
            video_config = self.config.get('video', {})
            fourcc_str = video_config.get('fourcc', 'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            fps = video_config.get('fps', 30)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Get input shape
        input_shape = tuple(self.model_config['model']['preset']['input_shape'][:2])
        detection_config = self.config.get('detection', {})
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                # Preprocess
                image_data = preprocess_image(image, input_shape)
                image_shape = (height, width)
                
                # Run inference
                predictions = self.model.predict(image_data, verbose=0)
                
                # Post-process
                nms_method = detection_config.get('nms_method', 'diou')
                use_wbf = detection_config.get('use_wbf', False)
                boxes, classes, scores = self.decoder.postprocess(
                    predictions,
                    image_shape,
                    input_shape,
                    max_boxes=detection_config.get('max_boxes', 100),
                    confidence=detection_config.get('confidence_threshold', 0.5),
                    nms_threshold=detection_config.get('nms_threshold', 0.45),
                    nms_method=nms_method,
                    use_wbf=use_wbf,
                    use_iol=detection_config.get('use_iol', True),
                    return_xyxy=True
                )
                
                # Draw boxes
                frame_rgb = np.array(image, dtype='uint8')
                annotated_frame = draw_boxes(
                    frame_rgb,
                    boxes,
                    classes,
                    scores,
                    self.class_names,
                    self.colors
                )
                
                # Convert back to BGR
                annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                # Write frame
                if output_path:
                    out.write(annotated_frame_bgr)
                
                # Display
                cv2.imshow('MultiGridDet', annotated_frame_bgr)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()
    
    def run(self):
        """Run inference based on input configuration."""
        input_config = self.config['input']
        output_config = self.config.get('output', {})
        
        input_type = input_config['type']
        source = input_config['source']
        
        print(f"\n🎯 Running inference:")
        print(f"   Input type: {input_type}")
        print(f"   Source: {source}\n")
        
        if input_type == 'image':
            # Single image inference
            annotated_image, boxes, classes, scores = self.predict_image(source)
            
            print(f"✓ Detected {len(boxes)} objects:")
            for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                class_name = self.class_names[cls]
                x1, y1, x2, y2 = box
                print(f"   {i+1}. {class_name}: {score:.3f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
            
            # Save result
            if output_config.get('save_result', True):
                output_dir = output_config.get('output_dir', 'output')
                os.makedirs(output_dir, exist_ok=True)
                
                output_path = os.path.join(output_dir, f"result_{Path(source).name}")
                result_image = Image.fromarray(annotated_image)
                result_image.save(output_path)
                print(f"\n✓ Result saved to: {output_path}")
            
            # Show result
            if output_config.get('show_result', True):
                result_image = Image.fromarray(annotated_image)
                result_image.show()
        
        elif input_type == 'video':
            # Video inference
            output_dir = output_config.get('output_dir', 'output')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"result_{Path(source).name}")
            
            self.predict_video(source, output_path if output_config.get('save_result', True) else None)
            print(f"✓ Result saved to: {output_path}")
        
        elif input_type == 'camera':
            # Camera inference
            camera_id = input_config.get('camera_id', 0)
            output_path = None
            
            if output_config.get('save_result', True):
                output_dir = output_config.get('output_dir', 'output')
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, "camera_output.mp4")
            
            self.predict_camera(camera_id, output_path)
        
        elif input_type == 'directory':
            # Process all images in directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = [
                f for f in Path(source).iterdir()
                if f.suffix.lower() in image_extensions
            ]
            
            print(f"Found {len(image_files)} images to process\n")
            
            output_dir = output_config.get('output_dir', 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            for i, image_file in enumerate(image_files, 1):
                print(f"[{i}/{len(image_files)}] Processing {image_file.name}...")
                
                annotated_image, boxes, classes, scores = self.predict_image(str(image_file))
                
                print(f"   Detected {len(boxes)} objects")
                
                # Save result
                if output_config.get('save_result', True):
                    output_path = os.path.join(output_dir, f"result_{image_file.name}")
                    result_image = Image.fromarray(annotated_image)
                    result_image.save(output_path)
            
            print(f"\n✓ Processed {len(image_files)} images")
            print(f"✓ Results saved to: {output_dir}")
        
        else:
            raise ValueError(f"Unknown input type: {input_type}")
        
        print("\n" + "=" * 80)
        print("✓ Inference Complete!")
        print("=" * 80)





