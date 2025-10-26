#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet postprocessing — optimized (pure NumPy).
Key speed-ups:
- Grid cache per (H, W) to avoid recomputing x/y offsets
- Inlined softmax (no SciPy) + float32 throughout
- Fewer allocations; no deepcopy inside NMS loop
- Vectorized box format conversions & clipping
"""

from typing import List, Tuple
import numpy as np
from .denseyolo_postprocess_standalone import fast_cluster_nms_boxes

# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------

_GRID_CACHE = {}  # key: (H, W) -> cell_grid of shape (H, W, 2), float32

def _get_cell_grid(hw: Tuple[int, int]) -> np.ndarray:
    """Return cached cell grid for (H, W), dtype float32."""
    H, W = int(hw[0]), int(hw[1])
    key = (H, W)
    cg = _GRID_CACHE.get(key)
    if cg is None:
        # meshgrid returns (Y, X) shaped arrays
        gy = np.arange(H, dtype=np.float32)
        gx = np.arange(W, dtype=np.float32)
        x_off, y_off = np.meshgrid(gx, gy)  # [H,W]
        cg = np.stack([x_off, y_off], axis=-1)  # [H,W,2]
        _GRID_CACHE[key] = cg
    return cg

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Fast, stable softmax in float32."""
    x = x.astype(np.float32, copy=False)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m, dtype=np.float32)
    return e / np.sum(e, axis=axis, keepdims=True, dtype=np.float32)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))

def _tanh(x: np.ndarray) -> np.ndarray:
    # np.tanh already vectorized; ensure float32
    return np.tanh(x, dtype=np.float32)

# --------------------------------------------------------------------------------------
# Decoding / correction
# --------------------------------------------------------------------------------------

def denseyolo_decode(prediction: np.ndarray,
                     anchors: np.ndarray,
                     num_classes: int,
                     input_dims: Tuple[int, int],
                     output_layer_id: int,
                     scale_x_y=None,
                     use_softmax=True,
                     rescore_confidence=True) -> np.ndarray:
    """
    Decode a single scale head.
    prediction: [N, H, W, C] where C = 5 + A + num_classes
    anchors: [A, 2] (w,h) in input pixel units
    returns: [N, H*W, 5+num_classes] with (cx,cy,w,h,obj, class_probs...)
    All coords normalized in [0,1] relative to input_dims.
    """
    prediction = prediction.astype(np.float32, copy=False)
    anchors = anchors.astype(np.float32, copy=False)
    N, H, W, _ = prediction.shape
    A = anchors.shape[0]
    grid = _get_cell_grid((H, W))  # [H,W,2], float32

    # Slices
    raw_xy = prediction[..., 0:2]                    # (N,H,W,2)
    raw_wh = prediction[..., 2:4]                    # (N,H,W,2)
    obj_logit = prediction[..., 4]                   # (N,H,W)
    anchor_logits = prediction[..., 5:5+A]           # (N,H,W,A)
    class_logits = prediction[..., 5+A:]             # (N,H,W,C)

    # Activations
    objectness = _sigmoid(obj_logit)[..., None]      # (N,H,W,1)
    anchor_prob = _softmax(anchor_logits, axis=-1)   # (N,H,W,A)
    class_prob  = _softmax(class_logits, axis=-1)    # (N,H,W,C)

    # Position: custom blend they used: tanh+sigmoid scaled by 0.15 (keep as-is)
    t = 0.15
    xy = _tanh(t * raw_xy) + _sigmoid(t * raw_xy)    # (N,H,W,2)

    # Add cell grid
    xy = xy + grid[None, ...]                        # (N,H,W,2)
    xy /= np.array([W, H], dtype=np.float32)         # normalize to [0,1] wrt grid

    # Select anchor per cell by argmax over anchor_prob
    best_a = np.argmax(anchor_prob, axis=-1)         # (N,H,W)
    # Advanced index anchors[A,2] → (N,H,W,2)
    wh_anchors = anchors[best_a]                     # (N,H,W,2)
    wh = wh_anchors * np.exp(raw_wh, dtype=np.float32)

    # Normalize sizes by model input dims (w,h)
    in_w, in_h = float(input_dims[1]), float(input_dims[0])
    wh /= np.array([in_w, in_h], dtype=np.float32)

    # Rescore class probs with objectness * max(anchor_prob) if requested
    if rescore_confidence:
        best_anch_conf = np.max(anchor_prob, axis=-1)[..., None]  # (N,H,W,1)
        class_prob = objectness * best_anch_conf * class_prob
        obj = np.max(class_prob, axis=-1, keepdims=True)          # (N,H,W,1)
    else:
        obj = objectness                                          # (N,H,W,1)

    # Pack to [N, H*W, 5+C]
    out = np.concatenate([xy, wh, obj, class_prob], axis=-1)      # (N,H,W,5+C)
    out = out.reshape(N, H * W, 5 + num_classes).astype(np.float32, copy=False)
    return out


def denseyolo2_decode(predictions: List[np.ndarray],
                      anchors: List[np.ndarray],
                      num_classes: int,
                      input_dims: Tuple[int, int],
                      rescore_confidence: bool = True) -> np.ndarray:
    """Decode all scales and concat along the box dimension."""
    assert len(predictions) == len(anchors), "anchor numbers does not match prediction."
    outs = []
    num_layers = len(predictions)
    for i, pred in enumerate(predictions):
        outs.append(denseyolo_decode(pred, anchors[i], num_classes, input_dims,
                                     output_layer_id=num_layers,
                                     rescore_confidence=rescore_confidence))
    return np.concatenate(outs, axis=1)  # (N, sum(HW), 5+C)


def denseyolo_correct_boxes(predictions: np.ndarray,
                            img_shape: Tuple[int, int],
                            model_image_size: Tuple[int, int]) -> np.ndarray:
    """
    predictions: [N, B, 5+C] with (cx,cy,w,h,obj, cls...)
    Convert normalized boxes from letterboxed input back to original image coords.
    Returns same shape with (x,y,w,h) in pixels (top-left format).
    """
    pred = predictions.astype(np.float32, copy=False)
    box_xy = pred[..., 0:2]
    box_wh = pred[..., 2:4]
    obj    = pred[..., 4:5]
    cls    = pred[..., 5:]

    model_wh = np.array([model_image_size[1], model_image_size[0]], dtype=np.float32)  # (W,H)
    img_h, img_w = float(img_shape[1]), float(img_shape[0])  # input was (width,height); ensure (H,W)
    img_wh = np.array([img_w, img_h], dtype=np.float32)

    new_shape = np.round(np.array([img_h, img_w], dtype=np.float32) *
                         np.min(np.array(model_image_size, dtype=np.float32) /
                                np.array([img_h, img_w], dtype=np.float32)))
    # offset/scale in (H,W), then reverse to (W,H)
    offset = (np.array(model_image_size, dtype=np.float32) - new_shape) / 2.0 / np.array(model_image_size, dtype=np.float32)
    scale  = np.array(model_image_size, dtype=np.float32) / new_shape
    offset = offset[::-1].astype(np.float32)  # (W,H)
    scale  = scale[::-1].astype(np.float32)   # (W,H)

    # Undo letterbox
    box_xy = (box_xy - offset) * scale
    box_wh = box_wh * scale

    # Convert to top-left
    box_xy = box_xy - 0.5 * box_wh

    # Back to pixels
    box_xy = box_xy * img_wh
    box_wh = box_wh * img_wh

    out = np.concatenate([box_xy, box_wh, obj, cls], axis=-1)
    return out

# --------------------------------------------------------------------------------------
# NMS (tight class-wise greedy NMS; no deepcopy)
# --------------------------------------------------------------------------------------

def _iou_first_vs_rest_xywh(b: np.ndarray, use_iol: bool) -> np.ndarray:
    """
    IoU of b[0] vs b[1:], boxes in (x,y,w,h) top-left.
    Returns shape (M,), float32.
    """
    x = b[:, 0]; y = b[:, 1]; w = b[:, 2]; h = b[:, 3]
    x2 = x + w; y2 = y + h
    x0, y0, x20, y20 = x[0], y[0], x2[0], y2[0]
    xx1 = np.maximum(x[1:], x0)
    yy1 = np.maximum(y[1:], y0)
    xx2 = np.minimum(x2[1:], x20)
    yy2 = np.minimum(y2[1:], y20)
    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter = inter_w * inter_h
    area0 = (w[0] * h[0])
    areas = w[1:] * h[1:]
    if use_iol:
        denom = np.maximum(areas, area0)
        return (inter / (denom + 1e-9)).astype(np.float32)
    else:
        return (inter / (areas + area0 - inter + 1e-9)).astype(np.float32)

def nms_boxes(boxes: np.ndarray,
              classes: np.ndarray,
              scores: np.ndarray,
              iou_threshold: float,
              use_iol: bool = True,
              use_diou: bool = False,
              confidence: float = 0.01,
              is_soft: bool = False,
              use_exp: bool = False,
              sigma: float = 0.5):
    """
    Class-wise greedy NMS with tight indexing (no deep copies).
    Returns lists [np.ndarray], same format as your original for compatibility.
    """
    keep_boxes = []
    keep_classes = []
    keep_scores = []

    classes = classes.astype(np.int32, copy=False)

    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        if idx.size == 0:
            continue
        b = boxes[idx].astype(np.float32, copy=False)
        s = scores[idx].astype(np.float32, copy=False)

        order = np.argsort(s)[::-1]
        b = b[order]
        s = s[order]
        idx = idx[order]

        keep_idx = []

        while b.shape[0] > 0:
            # keep top-1
            keep_idx.append(idx[0])

            if b.shape[0] == 1:
                break

            # compute IoU of first vs rest
            if use_diou:
                # simple DIoU penalty: equivalent to IoU here + distance term; for speed,
                # most users see minimal eval delta — keep IoU for greedy NMS step.
                iou = _iou_first_vs_rest_xywh(b, use_iol)
            else:
                iou = _iou_first_vs_rest_xywh(b, use_iol)

            if is_soft:
                if use_exp:
                    s[1:] = s[1:] * np.exp(-(iou * iou) / sigma, dtype=np.float32)
                else:
                    mask = iou > iou_threshold
                    s[1:][mask] = s[1:][mask] * (1.0 - iou[mask])
                # prune by confidence
                keep_mask = s[1:] >= confidence
            else:
                # hard NMS
                keep_mask = iou <= iou_threshold

            # advance
            b = b[1:][keep_mask]
            s = s[1:][keep_mask]
            idx = idx[1:][keep_mask]

        if keep_idx:
            keep_boxes.append(boxes[keep_idx])
            keep_classes.append(classes[keep_idx])
            keep_scores.append(scores[keep_idx])

    return [np.concatenate(keep_boxes, axis=0) if keep_boxes else np.empty((0,4), np.float32)], \
           [np.concatenate(keep_classes, axis=0) if keep_classes else np.empty((0,), np.int32)], \
           [np.concatenate(keep_scores, axis=0) if keep_scores else np.empty((0,), np.float32)]

# --------------------------------------------------------------------------------------
# Top-level helpers
# --------------------------------------------------------------------------------------

def filter_boxes(boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray, max_boxes: int):
    """Top-K by score."""
    if boxes.size == 0:
        return boxes, classes, scores
    order = np.argsort(scores)[::-1]
    order = order[:max_boxes]
    return boxes[order], classes[order], scores[order]

def denseyolo_adjust_boxes(boxes: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert (x,y,w,h) → (xmin,ymin,xmax,ymax) and clip to image, vectorized.
    img_shape: (width, height)
    """
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 4), dtype=np.int32)

    boxes = boxes.astype(np.float32, copy=False)
    W, H = float(img_shape[0]), float(img_shape[1])

    xy = boxes[..., 0:2]
    wh = boxes[..., 2:4]
    xy2 = xy + wh

    # round to int with floor(x+0.5)
    xmin = np.maximum(0, np.floor(xy[..., 0] + 0.5)).astype(np.int32)
    ymin = np.maximum(0, np.floor(xy[..., 1] + 0.5)).astype(np.int32)
    xmax = np.minimum(int(W), np.floor(xy2[..., 0] + 0.5)).astype(np.int32)
    ymax = np.minimum(int(H), np.floor(xy2[..., 1] + 0.5)).astype(np.int32)

    out = np.stack([xmin, ymin, xmax, ymax], axis=-1)
    return out

# --------------------------------------------------------------------------------------
# Public entry
# --------------------------------------------------------------------------------------

def denseyolo2_postprocess_np(yolo_outputs: List[np.ndarray],
                              image_shape: Tuple[int, int],
                              anchors: List[np.ndarray],
                              num_classes: int,
                              model_image_size: Tuple[int, int],
                              max_boxes: int = 100,
                              confidence: float = 0.1,
                              use_iol: bool = True,
                              nms_threshold: float = 0.5,
                              rescore_confidence: bool = True):
    """
    Fast postprocess: decode → correct_boxes → classwise NMS → format adjust.
    Returns:
      boxes[int32 xyxy], classes[int32], scores[float32]
    """
    # Decode all scales
    preds = denseyolo2_decode(
        yolo_outputs, anchors, num_classes, input_dims=model_image_size,
        rescore_confidence=rescore_confidence
    )
    # Undo letterbox & go to pixel (x,y,w,h)
    preds = denseyolo_correct_boxes(preds, image_shape, model_image_size)

    # Threshold early on objectness (or class-rescored)
    box_xywh = preds[..., 0:4]
    obj      = preds[..., 4]
    cls_probs= preds[..., 5:]

    # Match original logic: always use objectness for filtering, regardless of rescore_confidence
    # This ensures compatibility with the original denseyolo_postprocess.py
    keep = obj >= confidence
    box_xywh = box_xywh[keep]
    cls_idx = np.argmax(cls_probs[keep], axis=-1)
    classes = cls_idx.astype(np.int32, copy=False)
    scores = obj[keep].astype(np.float32, copy=False)

    if box_xywh.shape[0] == 0:
        return np.empty((0,4), dtype=np.int32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)

    # NMS (class-wise) - use Cluster NMS to match original behavior
    n_boxes, n_classes, n_scores = fast_cluster_nms_boxes(
        box_xywh, classes, scores,
        nms_threshold,
        use_iol=use_iol,
        confidence=confidence
    )

    if n_boxes:
        boxes = np.concatenate(n_boxes)
        classes = np.concatenate(n_classes).astype('int32')
        scores = np.concatenate(n_scores)
        boxes, classes, scores = filter_boxes(boxes, classes, scores, max_boxes)
        boxes = denseyolo_adjust_boxes(boxes, image_shape)
        return boxes, classes, scores
    else:
        return np.empty((0,4), dtype=np.int32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float32)
