
import numpy as np

def read_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert('RGB'))

def read_mask(path,threshold=.5):
    from PIL import Image
    image=Image.open(path)
    channel=image.getchannel('A') if 'A' in image.getbands() else image.convert('L')
    return np.asarray(channel,dtype=np.float32)/255 >= threshold

def boundary_edge_agreement(rgb,mask,tolerance=2,canny_low=100,canny_high=200):
    import cv2
    m=np.asarray(mask,bool)

    if not m.any(): return None
    eroded=cv2.erode(np.pad(m.astype(np.uint8),1),np.ones((3,3),np.uint8))[1:-1,1:-1]
    boundary=m & ~eroded.astype(bool)
    edges=cv2.Canny(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),canny_low,canny_high)>0
    if not edges.any(): return 0.
    distance=cv2.distanceTransform((~edges).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return float(np.mean(distance[boundary]<=tolerance))

def lpips_distance(rgb_a,rgb_b,model,mask=None):

    import torch
    import torch.nn.functional as F

    device=next(model.parameters()).device
    convert=lambda a:torch.from_numpy(np.array(a,copy=True)).permute(2,0,1).unsqueeze(0).float().to(device)/127.5-1
    model.eval()
    with torch.inference_mode():
        score=model(convert(rgb_a),convert(rgb_b))
        if mask is None: return float(score.mean())
        m=np.asarray(mask,bool)

        score=F.interpolate(score,size=m.shape,mode='bilinear',align_corners=False)
        weights=torch.from_numpy(m.astype(np.float32)).to(device)[None,None]
        return float((score*weights).sum()/weights.sum())

def object_measurements(instance_masks,depth,larger_is_near=True):


    objects={}
    for key,mask in instance_masks.items():
        mask=np.asarray(mask,bool)
        geometry=mask_geometry(mask)
        values=np.asarray(depth)[mask & np.isfinite(depth)]
        if geometry is None or not len(values): objects[key]=None; continue
        objects[key]={'center_xy':geometry['center'].tolist(),
            'depth':float(np.median(values))*(-1 if larger_is_near else 1)}
    return objects
