
import numpy as np
from .common import rgb_array, binary_mask, summarize_records
from .boundary_iou import inner_boundary


def boundary_edge_agreement(source, predicted_mask, tolerance=2.0,
                            canny_low=100, canny_high=200):
    import cv2
    if predicted_mask is None:
        return None
    image = rgb_array(source)
    mask = binary_mask(predicted_mask)

    contour = inner_boundary(mask,1)
    if not contour.any():
        return None
    gray = cv2.cvtColor(np.rint(image*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,canny_low,canny_high,L2gradient=True)>0
    if not edges.any():
        return 0.0
    distances = cv2.distanceTransform((~edges).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return float(np.mean(distances[contour] <= tolerance))


def summarize_edge_agreement(records, tool):
    return summarize_records(records,'Edge_agreement','T2',tool)
