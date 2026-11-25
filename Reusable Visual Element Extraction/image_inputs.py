
import numpy as np
from PIL import Image, ImageOps
from .common import rgb_array, alpha_array, binary_mask


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB'),dtype=np.float32)/255


def load_layer(path):

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        has_alpha = 'A' in image.getbands() or 'transparency' in image.info
        rgba = np.asarray(image.convert('RGBA'),dtype=np.float32)/255
        return rgba[...,:3].copy(), rgba[...,3].copy() if has_alpha else None


def load_mask(path):

    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert('L'),dtype=np.float32)/255


def soft_chroma_alpha(output_rgb, key_rgb, tau_low, tau_high):

    image = rgb_array(output_rgb)
    key = np.asarray(key_rgb,dtype=np.float32)

    distance = np.linalg.norm(image-key,axis=-1)
    return np.clip((distance-tau_low)/(tau_high-tau_low),0,1).astype(np.float32)


def choose_key_color(source_rgb, seed=42, grid_size=17):

    import cv2
    image=rgb_array(source_rgb)
    pixels=image.reshape(-1,3)
    rng=np.random.default_rng(seed)
    if len(pixels)>50000:
        pixels=pixels[rng.choice(len(pixels),50000,replace=False)]

    cv2.setRNGSeed(seed)
    _,_,palette=cv2.kmeans(pixels.astype(np.float32),16,None,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,1e-4),
        5,cv2.KMEANS_PP_CENTERS)
    axis=np.linspace(0,1,grid_size,dtype=np.float32)
    candidates=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    distances=((candidates[:,None,:]-palette[None,:,:])**2).sum(axis=-1).min(axis=1)
    return candidates[int(distances.argmax())]


def recover_foreground_rgb(keyed_rgb, estimated_alpha, key_rgb, epsilon=1e-4):

    image = rgb_array(keyed_rgb)
    alpha = alpha_array(estimated_alpha,image.shape[:2])[...,None]
    key = np.asarray(key_rgb,dtype=np.float32)

    foreground = np.divide(image-(1-alpha)*key,alpha,
                           out=np.zeros_like(image),where=alpha>epsilon)
    return np.clip(foreground,0,1)


def source_layer_from_mask(source_rgb, predicted_mask):

    image = rgb_array(source_rgb)
    return image.copy(), alpha_array(predicted_mask,image.shape[:2])


def visible_reference_alpha(target_alpha, occluder_alphas=()):

    visible = alpha_array(target_alpha).copy()
    for occluder in occluder_alphas:
        visible *= 1-alpha_array(occluder,visible.shape)
    return visible


def composite_on_background(foreground_rgb, alpha, background):
    image = rgb_array(foreground_rgb)
    a = alpha_array(alpha,image.shape[:2])[...,None]
    b = np.asarray(background,dtype=np.float32)
    if b.shape==(3,):
        b = np.broadcast_to(b,image.shape)
    b = rgb_array(b)

    return image*a+b*(1-a)


def resize_layer_to_source(foreground_rgb, alpha, source_shape):

    import cv2
    image = rgb_array(foreground_rgb)
    a = alpha_array(alpha,image.shape[:2])
    h,w = source_shape[:2]

    if image.shape[:2]==(h,w):
        return image.copy(),a.copy()
    interpolation = cv2.INTER_AREA if image.shape[0]>h else cv2.INTER_LINEAR
    premult = cv2.resize(image*a[...,None],(w,h),interpolation=interpolation)
    resized_alpha = cv2.resize(a,(w,h),interpolation=interpolation)
    rgb = np.divide(premult,resized_alpha[...,None],out=np.zeros_like(premult),
                    where=resized_alpha[...,None]>1e-6)
    return np.clip(rgb,0,1), np.clip(resized_alpha,0,1)


def align_layer_to_source(output_rgb, output_alpha, source_rgb,
                          source_target_mask=None, ratio=0.75,
                          min_inliers=6, reprojection_threshold=3.0):

    import cv2
    source = rgb_array(source_rgb)
    output = rgb_array(output_rgb)
    alpha = alpha_array(output_alpha,output.shape[:2])

    gray = lambda rgb: cv2.cvtColor(np.rint(rgb*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
    mask = (alpha>=0.5).astype(np.uint8)*255
    search = None
    if source_target_mask is not None:
        search = binary_mask(source_target_mask).astype(np.uint8)*255

    sift = cv2.SIFT_create()
    kp_o,desc_o = sift.detectAndCompute(gray(output),mask)
    kp_s,desc_s = sift.detectAndCompute(gray(source),search)
    if desc_o is None or desc_s is None or len(desc_s)<2:
        return None
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_o,desc_s,k=2)
    accepted = [pair[0] for pair in matches if len(pair)==2 and pair[0].distance<ratio*pair[1].distance]

    unique = {}
    for match in sorted(accepted,key=lambda m:m.distance):
        unique.setdefault(match.trainIdx,match)
    accepted = list(unique.values())
    if len(accepted)<min_inliers:
        return None
    src = np.float32([kp_o[m.queryIdx].pt for m in accepted])
    dst = np.float32([kp_s[m.trainIdx].pt for m in accepted])
    matrix,inliers = cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,
        ransacReprojThreshold=reprojection_threshold,maxIters=5000,confidence=0.99)
    if matrix is None or inliers is None or int(inliers.sum())<min_inliers or not np.isfinite(matrix).all():
        return None
    scale = float(np.hypot(matrix[0,0],matrix[1,0]))
    if scale<=1e-6:
        return None
    h,w = alpha.shape
    warped_a = cv2.warpAffine(alpha,matrix,(w,h),flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    warped_p = cv2.warpAffine(output*alpha[...,None],matrix,(w,h),flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    aligned_rgb = np.divide(warped_p,warped_a[...,None],out=np.zeros_like(warped_p),
                            where=warped_a[...,None]>1e-6)
    if not (warped_a>=0.5).any():
        return None
    center=np.array([w/2,h/2,1.0])
    return {'rgb':np.clip(aligned_rgb,0,1),'alpha':np.clip(warped_a,0,1),
            'matrix_output_to_source':matrix,
            'correction_scale':scale,'relative_scale_drift':abs(1/scale-1),
            'rotation_deg':float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0]))),
            'canvas_center_displacement_norm':float(np.linalg.norm(matrix@center-center[:2])/np.hypot(h,w)),
            'inliers':int(inliers.sum()),'matches':len(accepted)}


def target_crop(foreground_rgb, alpha, threshold=0.5):

    image=rgb_array(foreground_rgb)
    a=alpha_array(alpha,image.shape[:2])
    mask=binary_mask(a,threshold)
    if not mask.any():
        return None
    yy,xx=np.nonzero(mask)
    rgb=composite_on_background(image,a,[0.5,0.5,0.5])
    crop=rgb[yy.min():yy.max()+1,xx.min():xx.max()+1]
    pil=Image.fromarray(np.rint(crop*255).astype(np.uint8))
    side=max(pil.size)
    square=Image.new('RGB',(side,side),(128,128,128))
    square.paste(pil,((side-pil.width)//2,(side-pil.height)//2))
    return square


def sam_mask_from_box(source_rgb, box_xyxy, predictor):

    import torch
    image=rgb_array(source_rgb)
    box=np.asarray(box_xyxy,dtype=np.float32)
    h,w=image.shape[:2]

    with torch.inference_mode():
        predictor.set_image(np.rint(image*255).astype(np.uint8))
        masks,_,_=predictor.predict(point_coords=None,point_labels=None,
                                    box=box[None,:],multimask_output=False)
    return np.asarray(masks).reshape(-1,h,w)[0].astype(bool)
