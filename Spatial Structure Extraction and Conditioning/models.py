
import numpy as np

def grounding_boxes(image_tensor,caption,model,image_size,box_threshold=.35,text_threshold=.25):

    from groundingdino.util.inference import predict
    boxes,scores,phrases=predict(model=model,image=image_tensor,caption=caption,
        box_threshold=box_threshold,text_threshold=text_threshold,
        device=str(next(model.parameters()).device))
    boxes=boxes.detach().cpu().numpy()
    h,w=image_size
    center,size=boxes[:,:2],boxes[:,2:]
    xyxy=np.concatenate([center-size/2,center+size/2],axis=1)*[w,h,w,h]
    xyxy[:,[0,2]]=np.clip(xyxy[:,[0,2]],0,w)
    xyxy[:,[1,3]]=np.clip(xyxy[:,[1,3]],0,h)
    return xyxy,scores.detach().cpu().numpy(),phrases

def sam_from_boxes(rgb_uint8,boxes_xyxy,predictor):
    import torch
    with torch.inference_mode():
        predictor.set_image(np.asarray(rgb_uint8,dtype=np.uint8))
        masks,_,_=predictor.predict(point_coords=None,point_labels=None,
            box=np.asarray(boxes_xyxy,np.float32),multimask_output=False)
    return np.asarray(masks).reshape(-1,*rgb_uint8.shape[:2]).astype(bool)

def estimate_depth(rgb_uint8,depth_anything_model):

    import torch
    depth_anything_model.eval()
    with torch.inference_mode():
        return depth_anything_model.infer_image(np.asarray(rgb_uint8)[...,::-1].copy())

def clip_alignment(pil_image,prompt,model,preprocess,tokenize):
    import torch
    import torch.nn.functional as F
    device=next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        image=model.encode_image(preprocess(pil_image).unsqueeze(0).to(device)).float()
        text=model.encode_text(tokenize([prompt]).to(device)).float()
        return float(F.cosine_similarity(image,text).item())

def generate_condition(prompt,seed,generate_text,generate_mask,generate_depth,
                       generate_combined,condition,masks=None,depth=None):

    if condition=='Text only': return generate_text(prompt,seed)
    if condition=='Mask': return generate_mask(prompt,seed,masks)
    if condition=='Depth': return generate_depth(prompt,seed,depth)
    if condition in ('Mask + depth','Ground truth'): return generate_combined(prompt,seed,masks,depth)
    raise ValueError(condition)
