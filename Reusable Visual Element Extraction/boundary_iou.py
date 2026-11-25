import numpy as np
from .common import binary_mask, summarize_records


def inner_boundary(mask, iterations):
    import cv2
    padded = np.pad(mask.astype(np.uint8), 1)
    eroded = cv2.erode(padded, np.ones((3,3),np.uint8),
                      iterations=iterations, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask & ~eroded[1:-1,1:-1].astype(bool)


def boundary_iou(predicted_alpha, reference_alpha, ratio=0.02, threshold=0.5):
    reference = binary_mask(reference_alpha, threshold)
    if predicted_alpha is None:
        return None
    predicted = binary_mask(predicted_alpha, threshold)
    width = max(1, int(round(ratio * np.hypot(*reference.shape))))
    a,b = inner_boundary(predicted,width),inner_boundary(reference,width)
    return float((a & b).sum() / (a | b).sum())


