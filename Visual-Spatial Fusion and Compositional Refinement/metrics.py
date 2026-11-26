
import numpy as np

def geometry_errors(predicted_mask, reference_mask):

    p,g=np.asarray(predicted_mask,bool),np.asarray(reference_mask,bool)

    if not p.any(): return {'Center error':None,'Scale error':1.0}
    center=lambda m:np.array(np.nonzero(m)).mean(axis=1)
    return {'Center error':float(np.linalg.norm(center(p)-center(g))/np.hypot(*g.shape)),
            'Scale error':float(abs(np.sqrt(p.sum()/g.sum())-1))}

def luminance_seam(rgb,mask):

    x=np.asarray(rgb,float);m=np.asarray(mask,bool)
    if x.shape!=m.shape+(3,): raise ValueError('Canvas mismatch.')
    y=x@np.array([.2126,.7152,.0722])
    values=np.concatenate([abs(np.diff(y,axis=0))[m[1:]!=m[:-1]],
                           abs(np.diff(y,axis=1))[m[:,1:]!=m[:,:-1]]])
    return float(values.mean()) if len(values) else None

def nominal_cost(calls,extraction_calls=0):

    generation_calls=calls-extraction_calls
    return {'seconds':12*generation_calls+1.2*extraction_calls,
            'usd':.025*generation_calls+.002*extraction_calls}

def alpha_composite(foreground, background, alpha):
    f,b,a = map(lambda x: np.asarray(x,float),(foreground,background,alpha))

    return f*a[...,None]+b*(1-a[...,None])

def region_errors(output, reference, foreground_mask, protected_background_mask):
    x,y = np.asarray(output,float),np.asarray(reference,float)
    fg,bg = np.asarray(foreground_mask,bool),np.asarray(protected_background_mask,bool)

    mse = float(np.mean((x[fg]-y[fg])**2))
    return {'FG MSE':mse,'PSNR':float(-10*np.log10(mse)) if mse>0 else float('inf'),
            'BG MAE':float(np.mean(abs(x[bg]-y[bg])))}

def feature_similarity(source_features, output_features):

    a,b = np.asarray(source_features,float).ravel(),np.asarray(output_features,float).ravel()

    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))

def constraint_scores(binary_constraints):
    z=np.asarray(binary_constraints)

    return {'atomic_mean':float(z.mean()),'joint_success':int(z.prod())}

def select_incumbent(candidates, constraint_vectors):

    scores=[constraint_scores(z)['atomic_mean'] for z in constraint_vectors]
    index=int(np.argmax(scores))
    return candidates[index], index
