import numpy as np

def assign_boxes(gt_boxes, min_k=2, max_k=5):
    """assigning boxes to layers in a pyramid according to its area
    Params
    -----
    gt_boxes: of shape (N, 5), each entry is [x1, y1, x2, y2, cls]
    min_k: minimum pyramid layer id
    max_k: maximum pyramid layer id

    Returns
    -----
    k: of shape (N,), each entry is a id indicating the assigned layer id
    """
    k0 = 4
    if gt_boxes.size > 0:
        layer_ids = np.zeros((gt_boxes.shape[0], ), dtype=np.int32)
        ws = gt_boxes[:, 2] - gt_boxes[:, 0]
        hs = gt_boxes[:, 3] - gt_boxes[:, 1]
        areas = ws * hs
        k = np.floor(k0 + np.log2(np.sqrt(areas) / 224))
        inds = np.where(k < min_k)[0]
        k[inds] = min_k
        inds = np.where(k > max_k)[0]
        k[inds] = max_k
        return k.astype(np.int32)

    else:
        return np.asarray([], dtype=np.int32)
