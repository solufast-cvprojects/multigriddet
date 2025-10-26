#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Academic-quality visualizations for MultiGridDet evaluation.
Publication-ready plots commonly expected in computer vision papers.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Lazy imports for optional dependencies
def _import_plotting_libs():
    """Lazy import of plotting libraries."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        return plt, sns
    except ImportError as e:
        raise ImportError(
            "Visualization requires matplotlib and seaborn. "
            "Install with: pip install matplotlib seaborn"
        ) from e


def plot_precision_recall_curves(results: Dict[str, Any],
                                 predictions: List[Dict],
                                 ground_truths: List[Dict],
                                 class_names: List[str],
                                 config: Dict,
                                 save_dir: str) -> None:
    """
    Plot Precision-Recall curves for each class and averaged.
    
    Args:
        results: Evaluation results from calculate_map
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts
        class_names: List of class names
        config: PR curve configuration
        save_dir: Directory to save plots
    """
    plt, sns = _import_plotting_libs()
    from .metrics import match_predictions_to_gt, compute_precision_recall
    
    show_per_class = config.get('show_per_class', True)
    show_averaged = config.get('show_averaged', True)
    top_k = config.get('top_k', 10)
    style = config.get('style', 'paper')
    
    # Set style
    if style == 'paper':
        sns.set_style('whitegrid')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
    else:
        sns.set_style('darkgrid')
        plt.rcParams['font.size'] = 12
    
    pr_dir = os.path.join(save_dir, 'pr_curves')
    os.makedirs(pr_dir, exist_ok=True)
    
    # Get active classes from results
    per_class = results.get('per_class', {})
    if not per_class:
        print("[WARNING] No per-class results for PR curves")
        return
    
    # Sort classes by AP@0.5
    class_scores = []
    for class_name, class_results in per_class.items():
        ap50 = class_results.get('AP0.50', 0.0)
        class_scores.append((class_name, ap50))
    class_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Plot individual class PR curves
    if show_per_class:
        print(f"   Generating PR curves for top {top_k} classes...")
        for i, (class_name, ap50) in enumerate(class_scores[:top_k]):
            try:
                class_id = class_names.index(class_name) if class_name in class_names else int(class_name.split('_')[1])
                
                # Filter predictions and GTs for this class
                class_preds = [p for p in predictions if p['class'] == class_id]
                class_gts = [gt for gt in ground_truths if gt['class'] == class_id]
                
                if len(class_preds) == 0 or len(class_gts) == 0:
                    continue
                
                # Compute PR curve at IoU 0.5
                tp_flags, fp_flags, scores = match_predictions_to_gt(
                    class_preds, class_gts, iou_threshold=0.5
                )
                precisions, recalls = compute_precision_recall(tp_flags, fp_flags, len(class_gts))
                
                # Plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(recalls, precisions, 'b-', linewidth=2, label=f'PR Curve (AP={ap50:.3f})')
                ax.fill_between(recalls, precisions, alpha=0.2)
                ax.set_xlabel('Recall', fontsize=12)
                ax.set_ylabel('Precision', fontsize=12)
                ax.set_title(f'Precision-Recall Curve: {class_name}', fontsize=14, fontweight='bold')
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
                
                # Save
                save_path = os.path.join(pr_dir, f'pr_curve_{class_name.replace(" ", "_")}.png')
                plt.tight_layout()
                plt.savefig(save_path, dpi=config.get('dpi', 300), bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"[WARNING] Failed to plot PR curve for {class_name}: {e}")
                continue
    
    # Plot averaged PR curve
    if show_averaged:
        print("   Generating averaged PR curve...")
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot top K classes
            colors = plt.cm.tab20(np.linspace(0, 1, min(top_k, len(class_scores))))
            
            for idx, (class_name, ap50) in enumerate(class_scores[:top_k]):
                try:
                    class_id = class_names.index(class_name) if class_name in class_names else int(class_name.split('_')[1])
                    class_preds = [p for p in predictions if p['class'] == class_id]
                    class_gts = [gt for gt in ground_truths if gt['class'] == class_id]
                    
                    if len(class_preds) == 0 or len(class_gts) == 0:
                        continue
                    
                    tp_flags, fp_flags, scores = match_predictions_to_gt(
                        class_preds, class_gts, iou_threshold=0.5
                    )
                    precisions, recalls = compute_precision_recall(tp_flags, fp_flags, len(class_gts))
                    
                    ax.plot(recalls, precisions, color=colors[idx], linewidth=1.5, 
                           alpha=0.7, label=f'{class_name} (AP={ap50:.3f})')
                except:
                    continue
            
            ax.set_xlabel('Recall', fontsize=14)
            ax.set_ylabel('Precision', fontsize=14)
            ax.set_title(f'Precision-Recall Curves (Top {top_k} Classes)', fontsize=16, fontweight='bold')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8, ncol=2)
            
            save_path = os.path.join(pr_dir, 'pr_curve_averaged.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=config.get('dpi', 300), bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"[WARNING] Failed to plot averaged PR curve: {e}")
    
    print(f"   PR curves saved to: {pr_dir}")


def plot_confusion_matrix(predictions: List[Dict],
                         ground_truths: List[Dict],
                         class_names: List[str],
                         config: Dict,
                         save_dir: str) -> None:
    """
    Plot confusion matrix showing which classes are confused with each other.
    
    Args:
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts
        class_names: List of class names
        config: Confusion matrix configuration
        save_dir: Directory to save plot
    """
    plt, sns = _import_plotting_libs()
    from .metrics import match_predictions_to_gt
    
    normalize = config.get('normalize', True)
    top_k = config.get('top_k', 20)
    cmap = config.get('cmap', 'Blues')
    
    print(f"   Generating confusion matrix...")
    
    # Get active classes
    active_classes = sorted(set([p['class'] for p in predictions] + [gt['class'] for gt in ground_truths]))
    active_classes = active_classes[:top_k]  # Limit to top K
    
    if len(active_classes) == 0:
        print("[WARNING] No active classes for confusion matrix")
        return
    
    # Initialize confusion matrix
    n_classes = len(active_classes)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int32)
    
    # Map class IDs to matrix indices
    class_to_idx = {cls: idx for idx, cls in enumerate(active_classes)}
    
    # Match predictions to GTs and build confusion matrix
    for gt in ground_truths:
        gt_class = gt['class']
        if gt_class not in class_to_idx:
            continue
        
        gt_idx = class_to_idx[gt_class]
        gt_bbox = np.array(gt['bbox'])
        gt_image_id = gt['image_id']
        
        # Find predictions for same image
        image_preds = [p for p in predictions if p['image_id'] == gt_image_id]
        
        if len(image_preds) == 0:
            # Missed detection (no prediction)
            continue
        
        # Find best matching prediction
        best_iou = 0
        best_pred_class = None
        
        for pred in image_preds:
            pred_bbox = np.array(pred['bbox'])
            # Simple IoU calculation
            x1 = max(gt_bbox[0], pred_bbox[0])
            y1 = max(gt_bbox[1], pred_bbox[1])
            x2 = min(gt_bbox[2], pred_bbox[2])
            y2 = min(gt_bbox[3], pred_bbox[3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
                pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
                union = gt_area + pred_area - intersection
                iou = intersection / (union + 1e-6)
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred_class = pred['class']
        
        # If we have a match with IoU > 0.5, record it
        if best_iou > 0.5 and best_pred_class in class_to_idx:
            pred_idx = class_to_idx[best_pred_class]
            confusion[gt_idx, pred_idx] += 1
    
    # Normalize if requested
    if normalize:
        row_sums = confusion.sum(axis=1, keepdims=True)
        confusion = confusion.astype(np.float32) / (row_sums + 1e-6)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get class labels
    labels = [class_names[cls] if cls < len(class_names) else f'class_{cls}' 
              for cls in active_classes]
    
    sns.heatmap(confusion, annot=False, fmt='.2f' if normalize else 'd',
                cmap=cmap, xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                ax=ax)
    
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title(f'Confusion Matrix (Top {len(active_classes)} Classes)', 
                fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.get('dpi', 300), bbox_inches='tight')
    plt.close()
    
    print(f"   Confusion matrix saved to: {save_path}")


def plot_per_class_map(results: Dict[str, Any],
                       config: Dict,
                       save_dir: str) -> None:
    """
    Plot horizontal bar chart of AP per class.
    
    Args:
        results: Evaluation results from calculate_map
        config: Configuration dict
        save_dir: Directory to save plot
    """
    plt, sns = _import_plotting_libs()
    
    print("   Generating per-class mAP bar chart...")
    
    per_class = results.get('per_class', {})
    if not per_class:
        print("[WARNING] No per-class results for mAP bar chart")
        return
    
    # Extract class names and AP values
    classes = []
    ap50_values = []
    ap_values = []
    
    for class_name, class_results in per_class.items():
        classes.append(class_name)
        ap50_values.append(class_results.get('AP0.50', 0.0))
        ap_values.append(class_results.get('AP', 0.0))
    
    # Sort by AP@0.5
    sorted_indices = np.argsort(ap50_values)[::-1]
    classes = [classes[i] for i in sorted_indices]
    ap50_values = [ap50_values[i] for i in sorted_indices]
    ap_values = [ap_values[i] for i in sorted_indices]
    
    # Limit to top K
    top_k = min(20, len(classes))
    classes = classes[:top_k]
    ap50_values = ap50_values[:top_k]
    ap_values = ap_values[:top_k]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, max(8, len(classes) * 0.4)))
    
    y_pos = np.arange(len(classes))
    
    ax.barh(y_pos, ap50_values, align='center', alpha=0.7, label='AP@0.5')
    ax.barh(y_pos, ap_values, align='center', alpha=0.5, label='AP@0.5:0.95')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Average Precision', fontsize=12)
    ax.set_title(f'Per-Class Average Precision (Top {len(classes)} Classes)', 
                fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.legend(loc='best')
    ax.grid(axis='x', alpha=0.3)
    
    save_path = os.path.join(save_dir, 'per_class_map.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.get('dpi', 300), bbox_inches='tight')
    plt.close()
    
    print(f"   Per-class mAP chart saved to: {save_path}")


def plot_iou_distribution(predictions: List[Dict],
                          ground_truths: List[Dict],
                          config: Dict,
                          save_dir: str) -> None:
    """
    Plot distribution of IoU values for true positives.
    
    Args:
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts
        config: Configuration dict
        save_dir: Directory to save plot
    """
    plt, sns = _import_plotting_libs()
    
    print("   Generating IoU distribution histogram...")
    
    # Calculate IoU for all predictions
    ious = []
    
    for pred in predictions:
        pred_bbox = np.array(pred['bbox'])
        pred_class = pred['class']
        pred_image_id = pred['image_id']
        
        # Find matching GTs
        best_iou = 0
        for gt in ground_truths:
            if gt['class'] == pred_class and gt['image_id'] == pred_image_id:
                gt_bbox = np.array(gt['bbox'])
                
                # Calculate IoU
                x1 = max(pred_bbox[0], gt_bbox[0])
                y1 = max(pred_bbox[1], gt_bbox[1])
                x2 = min(pred_bbox[2], gt_bbox[2])
                y2 = min(pred_bbox[3], gt_bbox[3])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
                    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
                    union = pred_area + gt_area - intersection
                    iou = intersection / (union + 1e-6)
                    best_iou = max(best_iou, iou)
        
        if best_iou > 0:
            ious.append(best_iou)
    
    if len(ious) == 0:
        print("[WARNING] No IoU values to plot")
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(ious, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='IoU=0.5')
    ax.axvline(x=0.75, color='orange', linestyle='--', linewidth=2, label='IoU=0.75')
    
    ax.set_xlabel('IoU', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('IoU Distribution for Predictions', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics
    mean_iou = np.mean(ious)
    median_iou = np.median(ious)
    ax.text(0.02, 0.98, f'Mean IoU: {mean_iou:.3f}\nMedian IoU: {median_iou:.3f}',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    save_path = os.path.join(save_dir, 'iou_distribution.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.get('dpi', 300), bbox_inches='tight')
    plt.close()
    
    print(f"   IoU distribution saved to: {save_path}")


def plot_confidence_analysis(predictions: List[Dict],
                             ground_truths: List[Dict],
                             config: Dict,
                             save_dir: str) -> None:
    """
    Plot how precision, recall, and F1-score vary with confidence threshold.
    
    Args:
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts
        config: Configuration dict
        save_dir: Directory to save plot
    """
    plt, sns = _import_plotting_libs()
    from .metrics import match_predictions_to_gt, compute_precision_recall
    
    print("   Generating confidence threshold analysis...")
    
    # Test different confidence thresholds
    conf_thresholds = np.linspace(0, 1, 50)
    precisions = []
    recalls = []
    f1_scores = []
    
    for conf in conf_thresholds:
        # Filter predictions by confidence
        filtered_preds = [p for p in predictions if p['score'] >= conf]
        
        if len(filtered_preds) == 0:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
            continue
        
        # Match predictions to GTs
        tp_flags, fp_flags, scores = match_predictions_to_gt(
            filtered_preds, ground_truths, iou_threshold=0.5
        )
        
        # Compute precision and recall
        tp = np.sum(tp_flags)
        fp = np.sum(fp_flags)
        fn = len(ground_truths) - tp
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(conf_thresholds, precisions, 'b-', linewidth=2, label='Precision')
    ax.plot(conf_thresholds, recalls, 'r-', linewidth=2, label='Recall')
    ax.plot(conf_thresholds, f1_scores, 'g-', linewidth=2, label='F1-Score')
    
    # Find optimal F1 point
    best_f1_idx = np.argmax(f1_scores)
    best_conf = conf_thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    
    ax.axvline(x=best_conf, color='purple', linestyle='--', linewidth=1.5,
              label=f'Optimal (conf={best_conf:.2f}, F1={best_f1:.3f})')
    ax.scatter([best_conf], [best_f1], color='purple', s=100, zorder=5)
    
    ax.set_xlabel('Confidence Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision, Recall, and F1-Score vs Confidence Threshold', 
                fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, 'confidence_analysis.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.get('dpi', 300), bbox_inches='tight')
    plt.close()
    
    print(f"   Confidence analysis saved to: {save_path}")


def generate_evaluation_report(results: Dict[str, Any],
                               predictions: List[Dict],
                               ground_truths: List[Dict],
                               class_names: List[str],
                               config: Dict,
                               save_dir: str) -> None:
    """
    Master function to generate all evaluation plots based on configuration.
    
    Args:
        results: Evaluation results from calculate_map
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts
        class_names: List of class names
        config: Visualization configuration from YAML
        save_dir: Base directory to save all plots
    """
    print("\n" + "=" * 80)
    print("Generating Academic Evaluation Visualizations")
    print("=" * 80)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get output settings
    output_config = config.get('output', {})
    dpi = output_config.get('dpi', 300)
    
    # Get plot enables
    plots = config.get('plots', {})
    
    # Generate each plot if enabled
    if plots.get('precision_recall_curves', True):
        try:
            pr_config = config.get('pr_curves', {})
            pr_config['dpi'] = dpi
            plot_precision_recall_curves(results, predictions, ground_truths, 
                                        class_names, pr_config, save_dir)
        except Exception as e:
            print(f"[ERROR] Failed to generate PR curves: {e}")
    
    if plots.get('confusion_matrix', True):
        try:
            cm_config = config.get('confusion_matrix', {})
            cm_config['dpi'] = dpi
            plot_confusion_matrix(predictions, ground_truths, class_names, 
                                cm_config, save_dir)
        except Exception as e:
            print(f"[ERROR] Failed to generate confusion matrix: {e}")
    
    if plots.get('per_class_map_bar', True):
        try:
            plot_per_class_map(results, {'dpi': dpi}, save_dir)
        except Exception as e:
            print(f"[ERROR] Failed to generate per-class mAP chart: {e}")
    
    if plots.get('iou_distribution', True):
        try:
            plot_iou_distribution(predictions, ground_truths, {'dpi': dpi}, save_dir)
        except Exception as e:
            print(f"[ERROR] Failed to generate IoU distribution: {e}")
    
    if plots.get('confidence_analysis', True):
        try:
            plot_confidence_analysis(predictions, ground_truths, {'dpi': dpi}, save_dir)
        except Exception as e:
            print(f"[ERROR] Failed to generate confidence analysis: {e}")
    
    print("\n" + "=" * 80)
    print(f"All visualizations saved to: {save_dir}")
    print("=" * 80 + "\n")

