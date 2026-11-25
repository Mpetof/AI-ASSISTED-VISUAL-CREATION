
import numpy as np
from .common import rgb_array, binary_mask, tensor_image, summarize_records


def load_lpips(device='cuda'):
    import lpips
    return lpips.LPIPS(net='alex', spatial=True).to(device).eval()


def masked_lpips(output, source, region_mask, scorer, threshold=0.5):
    import torch
    import torch.nn.functional as F
    if output is None:
        return None
    x,y = rgb_array(output),rgb_array(source)
    mask = binary_mask(region_mask,threshold)

    if not mask.any():
        return None 
    x = np.where(mask[...,None],x,0.5).astype(np.float32)
    y = np.where(mask[...,None],y,0.5).astype(np.float32)
    device = next(scorer.parameters()).device
    scorer.eval()
    with torch.inference_mode():
        distance = scorer(tensor_image(x,device)*2-1,tensor_image(y,device)*2-1)

        distance = F.interpolate(distance,size=mask.shape,mode='bilinear',align_corners=False)
        weights = torch.as_tensor(mask,device=device,dtype=distance.dtype)[None,None]
        return float((distance*weights).sum().item()/weights.sum().item())


def summarize_masked_lpips(records, tier, tool):
    return summarize_records(records, 'LPIPS', tier, tool)
