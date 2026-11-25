
import numpy as np
from .common import binary_mask, summarize_records


def mask_iou(predicted_alpha, reference_alpha, threshold=0.5):
    reference = binary_mask(reference_alpha, threshold)

    if predicted_alpha is None:
        return None  # missing output, not an empty segmentation
    predicted = binary_mask(predicted_alpha, threshold)

    return float(np.logical_and(predicted, reference).sum() /
                 np.logical_or(predicted, reference).sum())


def summarize_iou(records, tool):
    return summarize_records(records, 'IoU', 'T1', tool)
