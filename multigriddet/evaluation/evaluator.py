#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Evaluator for MultiGridDet models.
- Uses tf.data to parallelize image IO & preprocessing
- Prefetches to overlap CPU pipeline with GPU inference
- Optional mixed precision (AMP) on GPU
- Parallel postprocess to avoid Python GIL bottleneck
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import ConfigLoader, build_model_for_inference
from ..utils.anchors import load_anchors, load_classes
from ..utils.tf_optimization import optimize_tf_gpu
from ..postprocess.denseyolo_postprocess import denseyolo2_postprocess_np
#from ..postprocess.multigriddet_postprocess import denseyolo2_postprocess_np 
from ..utils.preprocessing import preprocess_image, preprocess_image_batch
from PIL import Image

# ---- New: TensorFlow (used only for the fast path) ----
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


class MultiGridEvaluator:
    """Evaluator for MultiGridDet models (fast version)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.class_names = None
        self.anchors = None
        
        # GPU opts
        optimize_tf_gpu()
        
        # Mixed precision (optional; can be disabled via config)
        if TF_AVAILABLE and self.config.get("evaluation", {}).get("mixed_precision", True):
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
            except Exception:
                pass  # Safe to ignore if not supported

        # Load/merge configs
        model_config_path = config['model_config']
        self.model_config = ConfigLoader.load_config(model_config_path)
        self.full_config = ConfigLoader.merge_configs(self.model_config, config)
        
        print("=" * 80)
        print("MultiGridDet Evaluator Initialized (FAST)")
        print("=" * 80)
        
        self._load_model()
        
    def _load_model(self):
        print("\n🔨 Loading model...")
        
        weights_path = self.config.get('weights_path')
        if not weights_path:
            raise ValueError("weights_path not specified in config")
        
        classes_path = self.config['data']['classes_path']
        anchors_path = self.model_config['model']['preset']['anchors_path']
        
        self.class_names = load_classes(classes_path)
        self.anchors = load_anchors(anchors_path)
        
        print(f"   Classes: {len(self.class_names)}")
        print(f"   Anchors: {len(self.anchors)} scales")
        
        self.model = build_model_for_inference(self.full_config, weights_path)
        print(f" Model loaded successfully\n")
    
    # ---------- Annotation loading (unchanged logic) ----------
    def _load_annotations(self, annotation_file: str) -> List[Dict]:
        print(f"Loading annotations from {annotation_file}...")
        annotations = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                image_path = parts[0]
                boxes = []
                for box_str in parts[1:]:
                    box_parts = box_str.split(',')
                    if len(box_parts) == 5:
                        x1, y1, x2, y2, cls = map(float, box_parts)
                        boxes.append({'bbox': [x1, y1, x2, y2], 'class': int(cls)})
                annotations.append({'image_path': image_path, 'boxes': boxes})
        print(f" Loaded {len(annotations)} annotations\n")
        return annotations
    
    # ---------- Fast tf.data pipeline ----------
    def _build_dataset(self, annotations: List[Dict], input_shape: Tuple[int, int], batch_size: int):
        """
        Build a tf.data.Dataset that yields (batch_images, batch_meta)
        batch_meta: dict with 'image_ids' (int32) and 'orig_shapes' (int32 [H,W])
        """
        if not TF_AVAILABLE:
            return None  # caller will fallback

        AUTOTUNE = tf.data.AUTOTUNE

        # Prepare tensors for dataset (paths + ids + shapes_placeholder)
        paths = [a['image_path'] for a in annotations]
        image_ids = np.arange(len(annotations), dtype=np.int32)

        paths_ds = tf.data.Dataset.from_tensor_slices((paths, image_ids))

        # Python loader → wrapped for TF
        def _py_load_and_preprocess(path_bytes):
            path = path_bytes.decode('utf-8')
            # PIL load (fast path would be tf.io/decode_jpeg, but we reuse your preprocessing)
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_np = preprocess_image_batch(img, input_shape)  # HxWxC float32
            orig_shape = (img.size[1], img.size[0])  # (H, W)
            return img_np.astype(np.float32), np.array(orig_shape, dtype=np.int32)

        def _tf_map(path, img_id):
            image, orig_shape = tf.numpy_function(
                _py_load_and_preprocess, [path], [tf.float32, tf.int32]
            )
            # Set static shapes so Keras can build a fast graph
            image.set_shape((input_shape[0], input_shape[1], 3))
            orig_shape.set_shape((2,))
            return {
                "image": image,
                "image_id": img_id,
                "orig_shape": orig_shape
            }

        ds = (paths_ds
              .shuffle(min(4096, len(annotations)), reshuffle_each_iteration=False)
              .map(_tf_map, num_parallel_calls=AUTOTUNE)
              .batch(batch_size, drop_remainder=False)
              .prefetch(AUTOTUNE))
        return ds

    # ---------- Parallel postprocess ----------
    def _postprocess_batch_parallel(self,
                                    batch_raw_outputs,
                                    batch_meta,
                                    input_shape,
                                    confidence_threshold,
                                    nms_threshold,
                                    use_iol):
        """
        Postprocess each image in a batch in parallel threads.
        Returns: (predictions, ground_truths) lists
        """
        images_count = batch_meta["image_id"].shape[0]
        preds = []
        # Ground truths are supplied later from evaluate loop (kept outside)
        def _post_one(i):
            # Slice out ith image prediction(s)
            if isinstance(batch_raw_outputs, list):
                img_outputs = [o[i:i+1].numpy() for o in batch_raw_outputs]
            else:
                img_outputs = batch_raw_outputs[i:i+1].numpy()

            boxes, classes, scores = denseyolo2_postprocess_np(
                img_outputs,
                tuple(int(x) for x in batch_meta["orig_shape"][i].tolist()),
                self.anchors,
                len(self.class_names),
                input_shape,
                max_boxes=500,
                confidence=confidence_threshold,
                rescore_confidence=True,
                nms_threshold=nms_threshold,
                use_iol=use_iol
            )
            image_id = int(batch_meta["image_id"][i])
            # Convert to a compact ndarray to avoid Python dict overhead inside the thread
            if len(boxes) == 0:
                return []  # no preds
            out = []
            for b, c, s in zip(boxes, classes, scores):
                out.append((image_id, int(c), float(s), float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            return out

        max_workers = min(os.cpu_count() or 4, 8)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_post_one, i) for i in range(images_count)]
            for fu in as_completed(futures):
                res = fu.result()
                if res:
                    preds.extend(res)

        # Convert tuples → dicts (outside threads to keep overhead lower)
        predictions = [{
            'image_id': r[0],
            'class': r[1],
            'score': r[2],
            'bbox': [r[3], r[4], r[5], r[6]]
        } for r in preds]

        return predictions

    # ---------- Main evaluation ----------
    def evaluate(self) -> Dict[str, Any]:
        print("=" * 80)
        print("Starting Evaluation (FAST PATH)")
        print("=" * 80)
        
        annotation_file = self.config['data']['annotation']
        annotations = self._load_annotations(annotation_file)
        
        eval_config = self.config['evaluation']
        input_shape = tuple(eval_config['input_shape'][:2])
        confidence_threshold = eval_config['confidence_threshold']
        nms_threshold = eval_config['nms_threshold']
        use_iol = eval_config.get('use_iol', True)
        batch_size = int(eval_config.get('batch_size', 1))
        max_images = eval_config.get('max_images', None)
        use_tfdata = bool(eval_config.get('use_tfdata', True) and TF_AVAILABLE)
        
        print(f"[INFO] Evaluation Configuration:")
        print(f"   Input shape: {input_shape}")
        print(f"   Confidence threshold: {confidence_threshold}")
        print(f"   NMS threshold: {nms_threshold}")
        print(f"   Use IoL: {use_iol}")
        print(f"   Batch size: {batch_size}")
        print(f"   tf.data fast path: {use_tfdata and TF_AVAILABLE}")
        if max_images:
            print(f"   Max images: {max_images}")
        print()

        if max_images:
            annotations = annotations[:max_images]

        # GT cache by image_id (to avoid repeated per-image loops later)
        gt_by_img = {idx: a['boxes'] for idx, a in enumerate(annotations)}

        all_predictions: List[Dict] = []
        all_ground_truths: List[Dict] = []

        if use_tfdata:
            ds = self._build_dataset(annotations, input_shape, batch_size)
            total_images = len(annotations)
            processed = 0

            for batch in tqdm(ds, total= (total_images + batch_size - 1) // batch_size, desc="Batches"):
                # Prepare inputs
                batch_images = batch["image"]
                # Keras predict() handles prefetch nicely; no verbose to avoid host syncs
                batch_raw = self.model(batch_images, training=False)

                # Pull meta to numpy (small tensors only)
                meta = {
                    "image_id": batch["image_id"].numpy(),
                    "orig_shape": batch["orig_shape"].numpy()
                }

                # Parallel postprocess
                batch_predictions = self._postprocess_batch_parallel(
                    batch_raw, meta, input_shape, confidence_threshold, nms_threshold, use_iol
                )
                all_predictions.extend(batch_predictions)

                # Append GT for all images in this batch (vectorized)
                for img_id in meta["image_id"]:
                    for gt in gt_by_img[int(img_id)]:
                        all_ground_truths.append({
                            'image_id': int(img_id),
                            'class': gt['class'],
                            'bbox': gt['bbox']
                        })

                processed += meta["image_id"].shape[0]

            print(f"\n Processed {processed} images (tf.data fast path)")

        else:
            # Fallback to your previous batched-Numpy path (kept, but tidied)
            print("[WARNING] Falling back to legacy loader (consider enabling tf.data).")
            num_images = len(annotations)
            num_batches = (num_images + batch_size - 1) // batch_size

            for batch_idx in tqdm(range(num_batches), desc="Batches"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_images)
                batch_annotations = annotations[start_idx:end_idx]

                # Load and preprocess in parallel threads
                imgs, shapes, valid_idx = [], [], []
                def _load_one(i, annot):
                    try:
                        img = Image.open(annot['image_path'])
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img_np = preprocess_image_batch(img, input_shape)
                        return i, img_np, (img.size[1], img.size[0]), None
                    except Exception as e:
                        return i, None, None, e

                with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as ex:
                    futures = [ex.submit(_load_one, start_idx + i, a) for i, a in enumerate(batch_annotations)]
                    for fu in as_completed(futures):
                        i, img_np, hw, err = fu.result()
                        if err is None:
                            imgs.append(img_np)
                            shapes.append(hw)
                            valid_idx.append(i)

                if not imgs:
                    continue

                batch_in = np.stack(imgs, axis=0)  # NHWC
                raw = self.model.predict(batch_in, verbose=0)

                # Build meta arrays
                meta = {
                    "image_id": np.array(valid_idx, dtype=np.int32),
                    "orig_shape": np.array(shapes, dtype=np.int32)
                }

                batch_predictions = self._postprocess_batch_parallel(
                    raw, meta, input_shape, confidence_threshold, nms_threshold, use_iol
                )
                all_predictions.extend(batch_predictions)

                for img_id in meta["image_id"]:
                    for gt in gt_by_img[int(img_id)]:
                        all_ground_truths.append({
                            'image_id': int(img_id),
                            'class': gt['class'],
                            'bbox': gt['bbox']
                        })

            print(f"\n Processed {len(annotations)} images (legacy path)")

        print(f"   Total predictions: {len(all_predictions)}")
        print(f"   Total ground truths: {len(all_ground_truths)}\n")
        
        # --------- mAP calculation (unchanged external API) ----------
        print("[INFO] Calculating mAP metrics...")
        results = self._calculate_metrics(all_predictions, all_ground_truths, eval_config)
        
        if eval_config.get('save_results', True):
            self._save_results(results, eval_config)
        
        # --------- Generate visualizations if enabled ----------
        viz_config = self.config.get('visualizations', {})
        if viz_config.get('enabled', False):
            try:
                print("\n[INFO] Generating evaluation visualizations...")
                from .visualizations import generate_evaluation_report
                
                output_config = viz_config.get('output', {})
                viz_save_dir = output_config.get('save_dir', 'results/evaluation/plots')
                
                generate_evaluation_report(
                    results=results,
                    predictions=all_predictions,
                    ground_truths=all_ground_truths,
                    class_names=self.class_names,
                    config=viz_config,
                    save_dir=viz_save_dir
                )
            except ImportError as e:
                print(f"[WARNING] Visualization failed: {e}")
                print("         Install matplotlib and seaborn to enable visualizations:")
                print("         pip install matplotlib seaborn")
            except Exception as e:
                print(f"[WARNING] Visualization failed: {e}")
        
        return results
    
    # ---------- Metrics / Save / Print ----------
    def _process_single_image(self, annot: Dict, input_shape: Tuple[int, int],
                              confidence_threshold: float, nms_threshold: float,
                              use_iol: bool, image_id: int) -> Tuple[List[Dict], List[Dict]]:
        """Kept for error fallbacks; not used on fast path."""
        image_path = annot['image_path']
        predictions, ground_truths = [], []
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_data = preprocess_image(image, input_shape)
            image_shape = tuple(reversed(image.size))
            model_predictions = self.model.predict(image_data, verbose=0)
            boxes, classes, scores = denseyolo2_postprocess_np(
                model_predictions, image_shape, self.anchors, len(self.class_names),
                input_shape, max_boxes=500, confidence=confidence_threshold,
                rescore_confidence=True, nms_threshold=nms_threshold, use_iol=use_iol
            )
            for box, cls, score in zip(boxes, classes, scores):
                predictions.append({
                    'image_id': image_id,
                    'class': cls,
                    'bbox': box.tolist(),
                    'score': float(score)
                })
            for gt_box in annot['boxes']:
                ground_truths.append({
                    'image_id': image_id,
                    'class': gt_box['class'],
                    'bbox': gt_box['bbox']
                })
        except Exception as e:
            print(f"[WARNING] Error processing {image_path}: {e}")
        return predictions, ground_truths

    def _calculate_metrics(self, predictions: List, ground_truths: List, eval_config: Dict) -> Dict:
        from .metrics import calculate_map, print_map_results
        iou_thresholds = eval_config.get('iou_thresholds', None)
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        method = eval_config.get('interpolation_method', 'coco')

        print(f"   IoU thresholds: {iou_thresholds}")
        print(f"   Interpolation method: {method}")
        print(f"   Number of classes: {len(self.class_names)}")

        optimize_classes = eval_config.get('optimize_classes', True)
        cache_ious = eval_config.get('cache_ious', True)
        use_parallel = eval_config.get('use_parallel', True)

        print(f"   Optimizations: classes={optimize_classes}, cache={cache_ious}, parallel={use_parallel}")

        results = calculate_map(
            predictions=predictions,
            ground_truths=ground_truths,
            num_classes=len(self.class_names),
            iou_thresholds=iou_thresholds,
            class_names=self.class_names,
            method=method,
            optimize_classes=optimize_classes,
            cache_ious=cache_ious,
            use_parallel=use_parallel
        )

        results['evaluation_info'] = {
            'num_predictions': len(predictions),
            'num_ground_truths': len(ground_truths),
            'classes_predicted': len({p['class'] for p in predictions}),
            'classes_in_gt': len({gt['class'] for gt in ground_truths}),
            'iou_thresholds': iou_thresholds,
            'interpolation_method': method
        }
        return results
    
    def _save_results(self, results: Dict, eval_config: Dict):
        results_dir = eval_config.get('results_dir', 'results/evaluation')
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n Results saved to: {results_path}")
    
    def print_results(self, results: Dict):
        from .metrics import print_map_results
        print_map_results(results, top_k=10)
        if 'evaluation_info' in results:
            info = results['evaluation_info']
            print(f"\n[INFO] Evaluation Details:")
            print(f"   Total Predictions: {info['num_predictions']:,}")
            print(f"   Total Ground Truths: {info['num_ground_truths']:,}")
            print(f"   Classes Predicted: {info['classes_predicted']}")
            print(f"   Classes in GT: {info['classes_in_gt']}")
            print(f"   IoU Thresholds: {len(info['iou_thresholds'])} points")
            print(f"   Interpolation: {info['interpolation_method'].upper()}")
        print("\n" + "=" * 80)
