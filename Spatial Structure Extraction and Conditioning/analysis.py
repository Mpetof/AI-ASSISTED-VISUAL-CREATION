
import json
from pathlib import Path
from itertools import combinations
import numpy as np

def mean_ci(v,rng):
    return [float(np.mean(v)),*bootstrap_ci(v,rng)]

def compare(table,names,rng):
    pairs=[]
    for i,j in combinations(range(len(names)),2):
        d=table[:,i]-table[:,j]
        pairs.append({'a':names[i],'b':names[j],**wilcoxon_exact(d),
            'difference':mean_ci(d,rng),'rank_biserial':float(rank_biserial(d)),
            'effect_ci':bootstrap_ci(d,rng,effect=True)})
    for pair,p in zip(pairs,holm([p['p'] for p in pairs])): pair['p_holm']=p
    return {'n_scenes':len(table),'friedman':friedman_test(table),'pairs':pairs}

def analyze(runs,extraction,removal):
    rng=np.random.default_rng(11703)
    metrics=['Compliance','Ordinal','Position error','CLIP','Edit LPIPS']
    names=['Text only','Mask','Depth','Mask + depth','Ground truth']
    summary={};statistics=[];arrays={}
    for tier in ['T1','T2']:
        selected=[r for r in runs if r['tier']==tier]
        scenes=sorted({r['scene'] for r in selected});conditions=names if tier=='T1' else names[:4]
        table=np.empty((len(scenes),len(conditions),5))
        for i,scene in enumerate(scenes):
            for j,c in enumerate(conditions):
                rows=[r for r in selected if r['scene']==scene and r['condition']==c]
                if len(rows)!=5 or len({r['run'] for r in rows})!=5: raise ValueError('Incomplete/duplicate runs.')
                table[i,j]=np.mean([[r[m] for m in metrics] for r in rows],axis=0)
        arrays[tier]=table
        summary[tier]={c:{m:mean_ci(table[:,j,k],rng) for k,m in enumerate(metrics)} for j,c in enumerate(conditions)}
        if tier=='T1':
            for k,m in enumerate(metrics[:4]): statistics.append({'tier':tier,'metric':m,**compare(table[:,:,k],conditions,rng)})
    dev=extraction['T1-dev']; truth=np.asarray(dev['truth'])
    calibration={}
    for label,proxy,index in [('Boundary-edge vs boundary F','edge',1),('Judge ordinal vs true ordinal','judge',3),('Perturbation stability vs true ordinal','stability',3)]:
        rho=float(np.corrcoef(row_ranks(dev[proxy]),row_ranks(truth[:,index]))[0,1])
        calibration[label]={'rho':rho,'eligible':rho>=.7}
    removal_summary={}
    for label in ['Predicted','Oracle']:
        rows=[r for r in removal if r['mask']==label]
        values=[]
        for scene in sorted({r['scene'] for r in rows}):
            group=[r for r in rows if r['scene']==scene]
            if len(group)!=5 or len({r['run'] for r in group})!=5: raise ValueError('Invalid removal runs.')
            values.append(np.mean([r['values'] for r in group],axis=0))
        values=np.asarray(values)
        removal_summary[label+' mask']={m:mean_ci(values[:,j],rng) for j,m in enumerate(['Inside LPIPS','Boundary continuity','Outside MAE'])}

    a,b=arrays['T1'].mean(0)[:4,0],arrays['T2'].mean(0)[:,0]
    sa=np.array([np.sign(a[i]-a[j]) for i,j in combinations(range(4),2)])
    sb=np.array([np.sign(b[i]-b[j]) for i,j in combinations(range(4),2)])
    denom=np.sqrt(np.sum(sa!=0)*np.sum(sb!=0))
    return {'synthetic':True,'summary':summary,'statistics':statistics,
        'extraction_T2_proxy_means':{
            'Boundary-edge agreement':float(np.mean(extraction['T2']['edge'])),
            'Judge ordinal agreement':float(np.mean(extraction['T2']['judge']))},
        'T1_structure_edit_points':[
            {'condition':c,'edit_lpips':float(arrays['T1'][:,j,4].mean()),
             'position_error':float(arrays['T1'][:,j,2].mean())}
            for j,c in enumerate(names)],
        'proxy_calibration':calibration,'removal_summary':removal_summary,
        'shared_condition_compliance_tau':float((sa*sb).sum()/denom) if denom else None,
        'extraction_T1_mean':np.asarray(extraction['T1']['truth']).mean(0).tolist()}

def row_ranks(values):

    a=np.asarray(values,float)
    order=np.argsort(a,axis=-1,kind='stable')
    sorted_a=np.take_along_axis(a,order,axis=-1)
    n=a.shape[-1]
    idx=np.broadcast_to(np.arange(n),a.shape)
    first=np.concatenate([np.ones(a.shape[:-1]+(1,),bool),np.diff(sorted_a,axis=-1)!=0],axis=-1)
    last=np.concatenate([np.diff(sorted_a,axis=-1)!=0,np.ones(a.shape[:-1]+(1,),bool)],axis=-1)
    left=np.maximum.accumulate(np.where(first,idx,0),axis=-1)
    right=np.minimum.accumulate(np.where(last,idx,n-1)[...,::-1],axis=-1)[...,::-1]
    ranked=(left+right)/2+1
    result=np.empty_like(a)
    np.put_along_axis(result,order,ranked,axis=-1)
    return result


def rank_biserial(differences):

    d=np.round(np.asarray(differences,float),12)
    zeros=(d==0)
    ranks=row_ranks(abs(d))-zeros.sum(axis=-1,keepdims=True)
    ranks=np.where(zeros,0,ranks)
    numerator=(ranks*np.sign(d)).sum(axis=-1)
    denominator=ranks.sum(axis=-1)
    return np.divide(numerator,denominator,out=np.zeros_like(numerator,dtype=float),where=denominator>0)


def wilcoxon_exact(differences):

    d=np.round(np.asarray(differences,float),12)
    d=d[d!=0]
    if not len(d):
        return {'W':0.,'p':1.,'n_nonzero':0}
    ranks=np.rint(2*row_ranks(abs(d))).astype(int)
    positive=int(ranks[d>0].sum())
    total=int(ranks.sum())
    counts=[0]*(total+1)
    counts[0]=1
    reachable=0
    for rank in ranks:
        rank=int(rank)
        for index in range(reachable,-1,-1):
            counts[index+rank]+=counts[index]
        reachable+=rank
    tail=min(positive,total-positive)
    p=min(1.,2*sum(counts[:tail+1])/2**len(d))
    return {'W':tail/2,'p':p,'n_nonzero':len(d)}


def friedman_test(table):

    a=np.round(np.asarray(table,float),12)
    n,k=a.shape

    ranks=row_ranks(a)
    q=12/(n*k*(k+1))*np.square(ranks.sum(axis=0)).sum()-3*n*(k+1)
    ties=sum(sum(int(c)**3-int(c) for c in np.unique(row,return_counts=True)[1]) for row in a)
    correction=1-ties/(n*(k**3-k))
    if correction<=0:
        return {'statistic':0.,'df':k-1,'p':1.}
    q=max(0.,float(q/correction))
    x=q/2

    p=math.exp(-x)*(1+x) if k==5 else (math.exp(-x) if k==3 else math.erfc(math.sqrt(x))+2*math.sqrt(x/math.pi)*math.exp(-x))
    return {'statistic':q,'df':k-1,'p':min(1.,p)}


def bootstrap_ci(values,rng,n_boot=10000,effect=False):
    a=np.asarray(values,float)

    draws=a[rng.integers(len(a),size=(n_boot,len(a)))]
    estimates=rank_biserial(draws) if effect else draws.mean(axis=1)
    return np.quantile(estimates,[.025,.975]).tolist()


def holm(pvalues):
    p=np.asarray(pvalues,float)
    order=np.argsort(p)
    adjusted=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    result=np.empty_like(adjusted)
    result[order]=adjusted
    return result.tolist()




