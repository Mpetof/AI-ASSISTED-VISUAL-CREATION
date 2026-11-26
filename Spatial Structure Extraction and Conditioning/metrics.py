
import numpy as np

def mask_geometry(mask):
    mask=np.asarray(mask,bool)
    if mask.ndim!=2 or not mask.any():
        return None
    y,x=np.nonzero(mask)
    return {'center':np.array([x.mean(),y.mean()]),
            'box_xyxy':[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]}

def instance_metrics(predicted,reference,tolerance=2):

    import cv2
    p,g=np.asarray(predicted,bool),np.asarray(reference,bool)

    iou=float((p&g).sum()/(p|g).sum())
    if not p.any():
        return {'IoU':0.,'Boundary F':0.,'Center error':None,'missing_instance':True}
    def boundary(m):
        padded=np.pad(m.astype(np.uint8),1)
        eroded=cv2.erode(padded,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)[1:-1,1:-1]
        return m & ~eroded.astype(bool)
    bp,bg=boundary(p),boundary(g)
    distance=lambda b:cv2.distanceTransform((~b).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    precision=float(np.mean(distance(bg)[bp]<=tolerance))
    recall=float(np.mean(distance(bp)[bg]<=tolerance))
    f=2*precision*recall/(precision+recall) if precision+recall else 0.
    center=np.linalg.norm(mask_geometry(p)['center']-mask_geometry(g)['center'])/np.hypot(*g.shape)
    return {'IoU':iou,'Boundary F':f,'Center error':float(center),'missing_instance':False}

def depth_pairs(valid_mask,count=10000,seed=42):

    ids=np.flatnonzero(np.asarray(valid_mask,bool))

    rng=np.random.default_rng(seed)
    first=rng.integers(len(ids),size=count)
    second=(first+rng.integers(1,len(ids),size=count))%len(ids)
    return np.stack([ids[first],ids[second]],axis=1)

def ordinal_accuracy(predicted,reference,pairs,predicted_larger_is_near=False,
                     reference_tolerance=1e-3):

    p,g=np.asarray(predicted,float),np.asarray(reference,float)
    if p.shape!=g.shape: raise ValueError('Depth maps must share a canvas.')
    if predicted_larger_is_near: p=-p
    pairs=np.asarray(pairs,int)
    a,b=pairs[:,0],pairs[:,1]
    dp=p.ravel()[a]-p.ravel()[b]; dg=g.ravel()[a]-g.ravel()[b]
    valid=np.isfinite(dp)&np.isfinite(dg)&(abs(dg)>reference_tolerance)
    return {'accuracy':float(np.mean(np.sign(dp[valid])==np.sign(dg[valid]))) if valid.any() else None,
            'valid_pairs':int(valid.sum()),'excluded_pairs':int((~valid).sum())}

def aligned_absrel(predicted,metric_depth,valid_mask,inverse_depth_prediction=True):

    p,g=np.asarray(predicted,float),np.asarray(metric_depth,float)
    valid=np.asarray(valid_mask,bool)&np.isfinite(p)&np.isfinite(g)&(g>0)

    target=1/g[valid] if inverse_depth_prediction else g[valid]
    design=np.column_stack([p[valid],np.ones(valid.sum())])

    scale,shift=np.linalg.lstsq(design,target,rcond=None)[0]
    aligned=scale*p[valid]+shift
    if scale<=0 or np.any(aligned<=0): return {'AbsRel':None,'status':'invalid_alignment'}
    depth=1/aligned if inverse_depth_prediction else aligned
    return {'AbsRel':float(np.mean(abs(depth-g[valid])/g[valid])),
            'scale':float(scale),'shift':float(shift),'status':'ok'}

def relation_satisfaction(objects,relations):

    scores=[]
    for r in relations:
        a,b=objects.get(r['a']),objects.get(r['b'])
        if a is None or b is None: scores.append(False); continue
        margin=r.get('margin',0)
        kind=r['relation']
        if kind=='left_of': passed=a['center_xy'][0]+margin<b['center_xy'][0]
        elif kind=='above': passed=a['center_xy'][1]+margin<b['center_xy'][1]
        elif kind=='in_front_of': passed=a['depth']+margin<b['depth']
        else: raise ValueError('Unsupported relation.')
        scores.append(bool(passed))
    return {'Compliance':float(np.mean(scores)) if scores else None,'satisfied':scores}

def removal_errors(output,source,clean_reference,target_mask,boundary_tolerance=.05):

    import cv2
    x,s,g=map(lambda z:np.asarray(z,float),(output,source,clean_reference))
    m=np.asarray(target_mask,bool)
    if x.shape!=s.shape or x.shape!=g.shape or m.shape!=x.shape[:2] or not (~m).any():
        raise ValueError('Invalid image/mask shapes or empty outside region.')
    band=cv2.dilate(m.astype(np.uint8),np.ones((3,3),np.uint8))-cv2.erode(m.astype(np.uint8),np.ones((3,3),np.uint8))
    gray=lambda a:a@np.array([.2126,.7152,.0722])
    dx=np.stack(np.gradient(gray(x)),axis=-1); dg=np.stack(np.gradient(gray(g)),axis=-1)
    return {'Outside MAE':float(np.mean(abs(x[~m]-s[~m]))),
            'Boundary continuity':float(np.mean(np.linalg.norm(dx-dg,axis=-1)[band>0]<=boundary_tolerance)) if band.any() else None}
