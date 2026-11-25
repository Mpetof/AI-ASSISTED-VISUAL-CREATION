"""Record field: Swap_error. Five full-canvas LPIPS comparisons.

RGBA inputs use straight (not premultiplied) RGB. Composite both extracted
and reference layers over the exact same five fixed replacement backgrounds.
"""
import numpy as np
from .common import rgb_array, alpha_array, tensor_image, summarize_records


def swap_error(predicted_rgb, predicted_alpha, reference_rgb, reference_alpha,
               backgrounds, scorer):
    import torch
    if predicted_rgb is None or predicted_alpha is None:
        return None
    if len(backgrounds) != 5:
        raise ValueError('Exactly five replacement backgrounds are required.')
    p,r = rgb_array(predicted_rgb),rgb_array(reference_rgb)
    if p.shape != r.shape:
        raise ValueError('Foreground layers must share a canvas.')
    pa = alpha_array(predicted_alpha,p.shape[:2])[...,None]
    ra = alpha_array(reference_alpha,r.shape[:2])[...,None]
    if not ra.any():
        raise ValueError('Reference alpha must be nonempty.')
    device = next(scorer.parameters()).device
    scorer.eval()
    scores = []
    with torch.inference_mode():
        for image in backgrounds:
            b = rgb_array(image)
            if b.shape != p.shape:
                raise ValueError('Background must share foreground canvas.')
            a = p*pa + b*(1-pa)
            c = r*ra + b*(1-ra)
            value = scorer(tensor_image(a,device)*2-1,tensor_image(c,device)*2-1)
            scores.append(float(value.mean().item()))
    return float(np.mean(scores))


def summarize_swap_error(records, tool):
    return summarize_records(records, 'Swap_error', 'T1', tool)
