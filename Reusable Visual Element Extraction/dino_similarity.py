
import numpy as np
from .common import rgb_array, binary_mask, summarize_records


def load_dino(device='cuda', checkpoint='facebook/dinov2-base'):
    from transformers import AutoImageProcessor, AutoModel
    return (AutoImageProcessor.from_pretrained(checkpoint),
            AutoModel.from_pretrained(checkpoint).to(device).eval())


def dino_similarity(output, source, predicted_mask, processor, model):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    if output is None:
        return None
    x,y = rgb_array(output),rgb_array(source)
    mask = binary_mask(predicted_mask)

    if not mask.any():
        return None
    yy,xx = np.nonzero(mask)
    region = (slice(yy.min(),yy.max()+1),slice(xx.min(),xx.max()+1))
    images = []
    for rgb in (x,y):
        crop = np.where(mask[...,None],rgb,0.5)[region]
        image = Image.fromarray(np.rint(crop*255).astype(np.uint8))
        side = max(image.size)
        canvas = Image.new('RGB',(side,side),(128,128,128))
        canvas.paste(image,((side-image.width)//2,(side-image.height)//2))
        images.append(canvas)
    model.eval()
    inputs = processor(images=images,return_tensors='pt').to(next(model.parameters()).device)
    with torch.inference_mode():
        features = model(**inputs).last_hidden_state[:,0,:].float()
        if torch.any(features.norm(dim=-1)==0):
            return None
        return float(F.cosine_similarity(features[0:1],features[1:2]).item())


def summarize_dino(records, tool):
    return summarize_records(records,'DINO','T2',tool)
