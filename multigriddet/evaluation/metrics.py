#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation metrics for MultiGridDet models.
Implements mAP calculation with custom implementation (no pycocotools dependency).
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from ..utils.boxes import BoxUtils


def _get_safe_num_workers(num_tasks: int) -> int:
    """
    Determine a safe number of worker processes for multiprocessing.
    
    Caps the number of workers to avoid excessive memory usage on machines
    with many CPU cores, while still providing useful parallelism.
    """
    if num_tasks <= 0:
        return 1
    # Hard cap to keep memory usage under control on very wide machines.
    return max(1, min(cpu_count(), num_tasks, 8))


def calculate_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate IoU matrix between two sets of boxes using vectorized operations.
    
    Args:
        boxes1: First set of boxes (N, 4) in xyxy format
        boxes2: Second set of boxes (M, 4) in xyxy format
        
    Returns:
        IoU matrix (N, M) where [i,j] is IoU between boxes1[i] and boxes2[j]
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    
    if boxes1.shape[1] != 4 or boxes2.shape[1] != 4:
        raise ValueError("Boxes must have 4 coordinates")
    
    # Vectorized IoU calculation using broadcasting
    # boxes1: (N, 4), boxes2: (M, 4)
    # Expand dimensions for broadcasting: (N, 1, 4) and (1, M, 4)
    boxes1_expanded = boxes1[:, np.newaxis, :]  # (N, 1, 4)
    boxes2_expanded = boxes2[np.newaxis, :, :]  # (1, M, 4)
    
    # Calculate intersection coordinates
    x1 = np.maximum(boxes1_expanded[..., 0], boxes2_expanded[..., 0])  # max(x1, x1)
    y1 = np.maximum(boxes1_expanded[..., 1], boxes2_expanded[..., 1])  # max(y1, y1)
    x2 = np.minimum(boxes1_expanded[..., 2], boxes2_expanded[..., 2])  # min(x2, x2)
    y2 = np.minimum(boxes1_expanded[..., 3], boxes2_expanded[..., 3])  # min(y2, y2)
    
    # Calculate intersection area
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate areas of individual boxes
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    # Calculate union area
    union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
    
    # Calculate IoU, avoiding division by zero
    ious = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    
    return ious


def match_predictions_to_gt(predictions: List[Dict], 
                           ground_truths: List[Dict], 
                           iou_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match predictions to ground truths using IoU threshold.
    
    Args:
        predictions: List of prediction dicts with keys ['bbox', 'class', 'score', 'image_id']
        ground_truths: List of GT dicts with keys ['bbox', 'class', 'image_id']
        iou_threshold: IoU threshold for matching
        
    Returns:
        Tuple of (tp_flags, fp_flags, scores) where:
        - tp_flags: Boolean array indicating true positives
        - fp_flags: Boolean array indicating false positives  
        - scores: Confidence scores of predictions
    """
    if len(predictions) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Sort predictions by confidence (descending)
    pred_scores = np.array([p['score'] for p in predictions])
    sorted_indices = np.argsort(pred_scores)[::-1]
    
    # Initialize arrays
    tp_flags = np.zeros(len(predictions), dtype=bool)
    fp_flags = np.zeros(len(predictions), dtype=bool)
    scores = pred_scores[sorted_indices]
    
    # Track which GTs have been matched
    gt_matched = set()
    
    # For each prediction (in confidence order)
    for i, pred_idx in enumerate(sorted_indices):
        pred = predictions[pred_idx]
        pred_class = pred['class']
        pred_bbox = np.array(pred['bbox'])
        pred_image_id = pred['image_id']
        
        # Find matching GTs (same class, same image)
        matching_gts = []
        for gt_idx, gt in enumerate(ground_truths):
            if (gt['class'] == pred_class and 
                gt['image_id'] == pred_image_id and 
                gt_idx not in gt_matched):
                matching_gts.append((gt_idx, gt))
        
        if len(matching_gts) == 0:
            # No matching GTs - false positive
            fp_flags[i] = True
            continue
        
        # Calculate IoU with all matching GTs
        gt_bboxes = np.array([gt['bbox'] for _, gt in matching_gts])
        ious = []
        for gt_bbox in gt_bboxes:
            iou = BoxUtils.box_iou(pred_bbox, gt_bbox)
            ious.append(iou)
        
        # Find best match
        best_iou_idx = np.argmax(ious)
        best_iou = ious[best_iou_idx]
        best_gt_idx = matching_gts[best_iou_idx][0]
        
        if best_iou >= iou_threshold:
            # True positive
            tp_flags[i] = True
            gt_matched.add(best_gt_idx)
        else:
            # False positive
            fp_flags[i] = True
    
    return tp_flags, fp_flags, scores


def match_predictions_to_gt_cached(predictions: List[Dict], 
                                   ground_truths: List[Dict], 
                                   iou_threshold: float,
                                   iou_cache: Dict[Tuple[int, int], float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match predictions to ground truths using pre-computed IoU cache.
    
    Args:
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts
        iou_threshold: IoU threshold for matching
        iou_cache: Dictionary with pre-computed IoU values
        
    Returns:
        Tuple of (tp_flags, fp_flags, scores)
    """
    if len(predictions) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Sort predictions by confidence (descending)
    pred_scores = np.array([p['score'] for p in predictions])
    sorted_indices = np.argsort(pred_scores)[::-1]
    
    # Initialize arrays
    tp_flags = np.zeros(len(predictions), dtype=bool)
    fp_flags = np.zeros(len(predictions), dtype=bool)
    scores = pred_scores[sorted_indices]
    
    # Track which GTs have been matched
    gt_matched = set()
    
    # For each prediction (in confidence order)
    for i, pred_idx in enumerate(sorted_indices):
        pred = predictions[pred_idx]
        pred_class = pred['class']
        pred_image_id = pred['image_id']
        
        # Find matching GTs (same class, same image)
        matching_gts = []
        for gt_idx, gt in enumerate(ground_truths):
            if (gt['class'] == pred_class and 
                gt['image_id'] == pred_image_id and 
                gt_idx not in gt_matched):
                matching_gts.append(gt_idx)
        
        if len(matching_gts) == 0:
            # No matching GTs - false positive
            fp_flags[i] = True
            continue
        
        # Find best IoU from cache
        best_iou = 0.0
        best_gt_idx = None
        
        for gt_idx in matching_gts:
            cache_key = (pred_idx, gt_idx)
            if cache_key in iou_cache:
                iou = iou_cache[cache_key]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx is not None:
            # True positive
            tp_flags[i] = True
            gt_matched.add(best_gt_idx)
        else:
            # False positive
            fp_flags[i] = True
    
    return tp_flags, fp_flags, scores


def compute_precision_recall(tp_flags: np.ndarray, 
                           fp_flags: np.ndarray, 
                           num_gt: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute precision and recall from TP/FP flags.
    
    Args:
        tp_flags: True positive flags
        fp_flags: False positive flags
        num_gt: Number of ground truth objects
        
    Returns:
        Tuple of (precisions, recalls)
    """
    if len(tp_flags) == 0:
        return np.array([0.0]), np.array([0.0])
    
    # Cumulative TP and FP
    cum_tp = np.cumsum(tp_flags)
    cum_fp = np.cumsum(fp_flags)
    
    # Precision = TP / (TP + FP)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-8)
    
    # Recall = TP / (TP + FN) = TP / num_gt
    recalls = cum_tp / (num_gt + 1e-8)
    
    return precisions, recalls


def compute_average_precision(precisions: np.ndarray, 
                            recalls: np.ndarray, 
                            method: str = 'coco') -> float:
    """
    Compute Average Precision from precision-recall curve.
    
    Args:
        precisions: Precision values
        recalls: Recall values
        method: 'coco' or 'voc' interpolation method
        
    Returns:
        Average Precision value
    """
    if len(precisions) == 0 or len(recalls) == 0:
        return 0.0
    
    if method == 'voc':
        # VOC 11-point interpolation
        recall_thresholds = np.arange(0, 1.1, 0.1)
        precisions_interp = []
        
        for r in recall_thresholds:
            # Find precisions where recall >= r
            valid_precisions = precisions[recalls >= r]
            if len(valid_precisions) > 0:
                precisions_interp.append(np.max(valid_precisions))
            else:
                precisions_interp.append(0.0)
        
        return np.mean(precisions_interp)
    
    elif method == 'coco':
        # COCO all-point interpolation
        # Sort by recall
        sorted_indices = np.argsort(recalls)
        recalls_sorted = recalls[sorted_indices]
        precisions_sorted = precisions[sorted_indices]
        
        # Compute interpolated precision
        precisions_interp = np.zeros_like(precisions_sorted)
        for i in range(len(precisions_sorted)):
            precisions_interp[i] = np.max(precisions_sorted[i:])
        
        # Compute area under curve
        if len(recalls_sorted) > 1:
            ap = np.trapz(precisions_interp, recalls_sorted)
        else:
            ap = precisions_interp[0] * recalls_sorted[0]
        
        return ap
    
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_ap_for_class(predictions: List[Dict], 
                          ground_truths: List[Dict], 
                          class_id: int,
                          iou_threshold: float = 0.5,
                          method: str = 'coco') -> float:
    """
    Calculate Average Precision for a single class at a single IoU threshold.
    
    Args:
        predictions: All predictions
        ground_truths: All ground truths
        class_id: Class ID to evaluate
        iou_threshold: IoU threshold for matching
        method: Interpolation method ('coco' or 'voc')
        
    Returns:
        Average Precision for the class
    """
    # Filter by class
    class_predictions = [p for p in predictions if p['class'] == class_id]
    class_gts = [gt for gt in ground_truths if gt['class'] == class_id]
    
    if len(class_predictions) == 0:
        return 0.0 if len(class_gts) > 0 else 1.0
    
    if len(class_gts) == 0:
        return 0.0
    
    # Match predictions to GTs
    tp_flags, fp_flags, scores = match_predictions_to_gt(
        class_predictions, class_gts, iou_threshold
    )
    
    # Compute precision-recall
    precisions, recalls = compute_precision_recall(tp_flags, fp_flags, len(class_gts))
    
    # Compute AP
    ap = compute_average_precision(precisions, recalls, method)
    
    return ap


def calculate_ap_for_class_cached(predictions: List[Dict], 
                                  ground_truths: List[Dict], 
                                  class_id: int,
                                  iou_threshold: float,
                                  iou_cache: Dict[Tuple[int, int], float],
                                  method: str = 'coco') -> float:
    """
    Calculate Average Precision for a single class using IoU cache.
    
    Args:
        predictions: All predictions
        ground_truths: All ground truths
        class_id: Class ID to evaluate
        iou_threshold: IoU threshold for matching
        iou_cache: Pre-computed IoU cache
        method: Interpolation method ('coco' or 'voc')
        
    Returns:
        Average Precision for the class
    """
    # Filter by class
    class_predictions = [p for p in predictions if p['class'] == class_id]
    class_gts = [gt for gt in ground_truths if gt['class'] == class_id]
    
    if len(class_predictions) == 0:
        return 0.0 if len(class_gts) > 0 else 1.0
    
    if len(class_gts) == 0:
        return 0.0
    
    # Match predictions to GTs using cached IoU values
    tp_flags, fp_flags, scores = match_predictions_to_gt_cached(
        class_predictions, class_gts, iou_threshold, iou_cache
    )
    
    # Compute precision-recall
    precisions, recalls = compute_precision_recall(tp_flags, fp_flags, len(class_gts))
    
    # Compute AP
    ap = compute_average_precision(precisions, recalls, method)
    
    return ap


def get_active_classes(predictions: List[Dict], ground_truths: List[Dict], num_classes: int) -> List[int]:
    """
    Identify classes that appear in predictions or ground truths.
    
    Args:
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts  
        num_classes: Total number of classes
        
    Returns:
        List of class IDs that have at least one prediction or ground truth
    """
    pred_classes = set(p['class'] for p in predictions)
    gt_classes = set(gt['class'] for gt in ground_truths)
    active_classes = sorted(pred_classes | gt_classes)
    
    return active_classes


def compute_iou_cache_for_class(predictions: List[Dict], ground_truths: List[Dict], class_id: int) -> Dict[Tuple[int, int], float]:
    """
    Pre-compute IoU matrix for a specific class to avoid recalculation.
    
    Args:
        predictions: All predictions
        ground_truths: All ground truths
        class_id: Class ID to compute cache for
        
    Returns:
        Dictionary mapping (pred_idx, gt_idx) -> IoU value
    """
    # Filter by class
    class_predictions = [p for p in predictions if p['class'] == class_id]
    class_gts = [gt for gt in ground_truths if gt['class'] == class_id]
    
    if len(class_predictions) == 0 or len(class_gts) == 0:
        return {}
    
    # Create IoU cache
    iou_cache = {}
    
    # Group predictions and GTs by image_id for efficiency
    pred_by_image = {}
    gt_by_image = {}
    
    for i, pred in enumerate(class_predictions):
        image_id = pred['image_id']
        if image_id not in pred_by_image:
            pred_by_image[image_id] = []
        pred_by_image[image_id].append((i, pred))
    
    for j, gt in enumerate(class_gts):
        image_id = gt['image_id']
        if image_id not in gt_by_image:
            gt_by_image[image_id] = []
        gt_by_image[image_id].append((j, gt))
    
    # Compute IoU only for predictions and GTs in the same image
    for image_id in pred_by_image:
        if image_id not in gt_by_image:
            continue
            
        image_preds = pred_by_image[image_id]
        image_gts = gt_by_image[image_id]
        
        # Extract bounding boxes
        pred_boxes = np.array([pred['bbox'] for _, pred in image_preds])
        gt_boxes = np.array([gt['bbox'] for _, gt in image_gts])
        
        # Compute IoU matrix using vectorized operation
        iou_matrix = calculate_iou_matrix(pred_boxes, gt_boxes)
        
        # Store in cache with original indices
        for i, (pred_idx, _) in enumerate(image_preds):
            for j, (gt_idx, _) in enumerate(image_gts):
                iou_cache[(pred_idx, gt_idx)] = float(iou_matrix[i, j])
    
    return iou_cache


def _calculate_ap_worker(args):
    """Worker function for parallel AP calculation."""
    predictions, ground_truths, class_id, iou_threshold, method = args
    return calculate_ap_for_class(predictions, ground_truths, class_id, iou_threshold, method)


def _calculate_ap_worker_cached(args):
    """Worker function for parallel AP calculation with IoU cache."""
    predictions, ground_truths, class_id, iou_threshold, iou_cache, method = args
    return calculate_ap_for_class_cached(predictions, ground_truths, class_id, iou_threshold, iou_cache, method)


def calculate_map(predictions: List[Dict], 
                 ground_truths: List[Dict], 
                 num_classes: int,
                 iou_thresholds: List[float] = None,
                 class_names: List[str] = None,
                 method: str = 'coco',
                 use_parallel: bool = True,
                 optimize_classes: bool = True,
                 cache_ious: bool = True) -> Dict[str, Any]:
    """
    Calculate mAP across classes and IoU thresholds.
    
    Args:
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts
        num_classes: Number of classes
        iou_thresholds: List of IoU thresholds to evaluate
        class_names: List of class names
        method: Interpolation method ('coco' or 'voc')
        
    Returns:
        Dictionary containing mAP results
    """
    if iou_thresholds is None:
        # Default COCO thresholds
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]
    
    results = {
        'mAP': 0.0,  # mAP@0.5:0.95
        'mAP50': 0.0,  # mAP@0.5
        'mAP75': 0.0,  # mAP@0.75
        'per_class': {},
        'per_iou': {},
        'num_predictions': len(predictions),
        'num_ground_truths': len(ground_truths)
    }
    
    # Determine which classes to process
    if optimize_classes:
        active_classes = get_active_classes(predictions, ground_truths, num_classes)
        print(f"📊 Active classes: {len(active_classes)}/{num_classes}")
    else:
        active_classes = list(range(num_classes))
    
    # Calculate AP for each class at each IoU threshold
    class_aps = {}
    iou_aps = {iou: [] for iou in iou_thresholds}
    
    total_tasks = len(active_classes) * len(iou_thresholds)
    
    if use_parallel and len(active_classes) > 1:
        try:
            # Parallel processing for multiple classes
            num_workers = _get_safe_num_workers(len(active_classes))
            print(f"🚀 Using parallel processing with {num_workers} cores...")
            
            # For large datasets, disable IoU caching as it becomes too expensive
            num_predictions = len(predictions)
            if cache_ious and num_predictions > 10000:
                print(f"⚠️  Large dataset detected ({num_predictions} predictions). Disabling IoU caching for performance.")
                cache_ious = False
            
            if cache_ious:
                # Parallel processing with IoU caching (only for small datasets)
                print("🔄 Computing IoU caches for all classes...")
                iou_caches = {}
                for class_id in active_classes:
                    print(f"   Computing cache for class {class_id}...")
                    iou_caches[class_id] = compute_iou_cache_for_class(predictions, ground_truths, class_id)
                
                # Create tasks for parallel processing with cached IoU
                tasks = []
                for class_id in active_classes:
                    for iou_threshold in iou_thresholds:
                        tasks.append((predictions, ground_truths, class_id, iou_threshold, iou_caches[class_id], method))
                
                # Process in parallel with cached IoU
                print(f"   Processing {len(tasks)} tasks (classes × IoU thresholds)...")
                with Pool(processes=_get_safe_num_workers(len(tasks))) as pool:
                    ap_results = list(tqdm(
                        pool.imap(_calculate_ap_worker_cached, tasks),
                        total=len(tasks),
                        desc="   Computing AP",
                        unit="task"
                    ))
            else:
                # Parallel processing without caching (faster for large datasets)
                print("⚡ Using optimized parallel processing (no IoU caching)...")
                tasks = []
                for class_id in active_classes:
                    for iou_threshold in iou_thresholds:
                        tasks.append((predictions, ground_truths, class_id, iou_threshold, method))
                
                # Process in parallel
                print(f"   Processing {len(tasks)} tasks (classes × IoU thresholds)...")
                with Pool(processes=_get_safe_num_workers(len(tasks))) as pool:
                    ap_results = list(tqdm(
                        pool.imap(_calculate_ap_worker, tasks),
                        total=len(tasks),
                        desc="   Computing AP",
                        unit="task"
                    ))
            
            # Organize results
            result_idx = 0
            for class_id in active_classes:
                class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
                class_ap_results = {}
                
                for iou_threshold in iou_thresholds:
                    ap = ap_results[result_idx]
                    class_ap_results[f'AP{iou_threshold:.2f}'] = ap
                    iou_aps[iou_threshold].append(ap)
                    result_idx += 1
                
                # Average AP across IoU thresholds for this class
                class_ap_avg = np.mean(list(class_ap_results.values()))
                class_ap_results['AP'] = class_ap_avg
                class_aps[class_name] = class_ap_results
        except OSError as e:
            # If the system cannot allocate memory for worker processes, fall back
            # to a safe sequential evaluation path instead of crashing.
            print(f"[WARNING] Parallel mAP evaluation failed: {e}")
            print("          Falling back to sequential evaluation without multiprocessing.")
            return calculate_map(
                predictions=predictions,
                ground_truths=ground_truths,
                num_classes=num_classes,
                iou_thresholds=iou_thresholds,
                class_names=class_names,
                method=method,
                use_parallel=False,
                optimize_classes=optimize_classes,
                cache_ious=False,
            )
    else:
        # Sequential processing (fallback or single class)
        print(f"   Processing {len(active_classes)} classes sequentially...")
        class_pbar = tqdm(active_classes, desc="   Classes", unit="class")
        for class_id in class_pbar:
            class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
            class_pbar.set_postfix({"current": class_name})
            class_ap_results = {}
            
            if cache_ious:
                # Compute IoU cache once per class
                iou_cache = compute_iou_cache_for_class(predictions, ground_truths, class_id)
                
                # Reuse cache for all IoU thresholds
                for iou_threshold in iou_thresholds:
                    ap = calculate_ap_for_class_cached(
                        predictions, ground_truths, class_id, iou_threshold, iou_cache, method
                    )
                    class_ap_results[f'AP{iou_threshold:.2f}'] = ap
                    iou_aps[iou_threshold].append(ap)
            else:
                # Original non-cached version
                for iou_threshold in iou_thresholds:
                    ap = calculate_ap_for_class(
                        predictions, ground_truths, class_id, iou_threshold, method
                    )
                    class_ap_results[f'AP{iou_threshold:.2f}'] = ap
                    iou_aps[iou_threshold].append(ap)
            
            # Average AP across IoU thresholds for this class
            class_ap_avg = np.mean(list(class_ap_results.values()))
            class_ap_results['AP'] = class_ap_avg
            class_aps[class_name] = class_ap_results
    
    results['per_class'] = class_aps
    
    # Calculate mAP for each IoU threshold
    for iou_threshold in iou_thresholds:
        if len(iou_aps[iou_threshold]) > 0:
            map_iou = np.mean(iou_aps[iou_threshold])
            results['per_iou'][f'mAP{iou_threshold:.2f}'] = map_iou
    
    # Calculate overall mAP metrics
    if 0.5 in iou_thresholds:
        results['mAP50'] = results['per_iou'].get('mAP0.50', 0.0)
    
    if 0.75 in iou_thresholds:
        results['mAP75'] = results['per_iou'].get('mAP0.75', 0.0)
    
    # mAP@0.5:0.95 (average across all IoU thresholds)
    if len(iou_thresholds) > 0:
        results['mAP'] = np.mean([results['per_iou'].get(f'mAP{iou:.2f}', 0.0) 
                                 for iou in iou_thresholds])
    
    return results


def print_map_results(results: Dict[str, Any], top_k: int = 10):
    """
    Print mAP results in a formatted way.
    
    Args:
        results: Results dictionary from calculate_map
        top_k: Number of top classes to show
    """
    print("\n" + "=" * 80)
    print("mAP Evaluation Results")
    print("=" * 80)
    
    print(f"\n📊 Overall Metrics:")
    print(f"   mAP@0.5:0.95: {results['mAP']:.4f}")
    print(f"   mAP@0.5:      {results['mAP50']:.4f}")
    print(f"   mAP@0.75:     {results['mAP75']:.4f}")
    
    print(f"\n📈 Per-IoU mAP:")
    for iou, map_val in results['per_iou'].items():
        print(f"   {iou}: {map_val:.4f}")
    
    print(f"\n📋 Per-Class Results (Top {top_k} by AP@0.5):")
    per_class = results['per_class']
    
    # Sort classes by AP@0.5
    class_scores = []
    for class_name, class_results in per_class.items():
        ap50 = class_results.get('AP0.50', 0.0)
        class_scores.append((class_name, ap50, class_results))
    
    class_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (class_name, ap50, class_results) in enumerate(class_scores[:top_k]):
        ap = class_results.get('AP', 0.0)
        print(f"   {i+1:2d}. {class_name:15s}: AP@0.5={ap50:.4f}, AP={ap:.4f}")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total Predictions: {results['num_predictions']:,}")
    print(f"   Total Ground Truths: {results['num_ground_truths']:,}")
    
    print("\n" + "=" * 80)
