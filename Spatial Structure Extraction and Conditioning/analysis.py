
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


