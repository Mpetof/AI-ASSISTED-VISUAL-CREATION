
import math
from itertools import combinations
import numpy as np
from .common import summarize_records

METRICS = {
    'T1': ['IoU','Boundary_IoU','LPIPS','Swap_error','Usable'],
    'T2': ['LPIPS','DINO','Edge_agreement','Identity'],
}
TOOLS = {
    'T1': ['SAM 2 (oracle box)','Grounded SAM 2','Gemini (scenario)','Doubao (scenario)'],
    'T2': ['Grounded SAM 2','Gemini (scenario)','Doubao (scenario)'],
}


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

    p=math.exp(-x) if k==3 else math.erfc(math.sqrt(x))+2*math.sqrt(x/math.pi)*math.exp(-x)
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


def analyze_extraction(records,n_boot=10000,seed=20261102,alpha=.05):

    if n_boot<100 or not 0<alpha<1:
        raise ValueError('Invalid bootstrap count or alpha.')
    records=[dict(r) for r in records]
    for r in records:
        if r['tier'] not in TOOLS or r['tool'] not in TOOLS[r['tier']]:
            raise ValueError('Unexpected tier/tool label.')
        if r.get('no_output',False):
            for m in METRICS[r['tier']]:
                r[m]=0 if m in ('Usable','Identity') else None
        for m in ('Usable','Identity'):
            if m in METRICS[r['tier']] and r.get(m) not in (0,1):
                raise ValueError('Binary outcomes must be recorded for every run.')
    rng=np.random.default_rng(seed)
    output={'metadata':{'data_status':'analysis_of_supplied_simulated_scores',
        'seed':seed,'bootstrap_resamples':n_boot,'interval':'percentile 95%, unadjusted',
        'friedman':'tie-corrected chi-square approximation',
        'wilcoxon':'two-sided exact conditional sign distribution, average ties, omit zeros',
        'holm_family':'tier x metric','effect_sign':'first minus second in raw metric units',
        'all_zero_effect_convention':0},'summary':[],'statistics':[],
        'category_iou':[],'failures':[]}
    for tier,metrics in METRICS.items():
        tools=TOOLS[tier]
        for metric in metrics:
            summaries={tool:summarize_records(records,metric,tier,tool) for tool in tools}
            all_tasks=sorted({r['task'] for r in records if r['tier']==tier})
            common=sorted(set.intersection(*(set(s['task_means']) for s in summaries.values())))
            for tool,s in summaries.items():
                lo,hi=bootstrap_ci(list(s['task_means'].values()),rng,n_boot)
                output['summary'].append({'tier':tier,'tool':tool,'metric':metric,
                    'mean':s['mean'],'low':lo,'high':hi,
                    **{k:v for k,v in s.items() if k.startswith('n_')}})
            omnibus=friedman_test([[summaries[t]['task_means'][i] for t in tools] for i in common])
            pairs=[]
            for a,b in combinations(tools,2):
                x,y=summaries[a]['task_means'],summaries[b]['task_means']
                paired=sorted(x.keys()&y.keys())
                d=np.array([x[i]-y[i] for i in paired])
                if len(d)<2:
                    raise ValueError('Insufficient paired tasks; do not silently change the testing family.')
                test=wilcoxon_exact(d)
                pairs.append({'a':a,'b':b,'n_tasks':len(d),'n_excluded':len(all_tasks)-len(d),
                    'mean_difference':float(d.mean()),'difference_ci95':bootstrap_ci(d,rng,n_boot),
                    'rank_biserial':float(rank_biserial(d)),
                    'effect_ci95':bootstrap_ci(d,rng,n_boot,effect=True),**test})
            for pair,p in zip(pairs,holm([r['p'] for r in pairs])):
                pair['p_holm']=p
                pair['reject_holm']=p<alpha
                pair['posthoc_supported']=p<alpha and omnibus['p']<alpha
            output['statistics'].append({'tier':tier,'metric':metric,
                'n_friedman_tasks':len(common),'n_friedman_excluded':len(all_tasks)-len(common),
                'friedman':omnibus,'pairs':pairs})
        selected=[r for r in records if r['tier']==tier]
        for tool in tools:
            rows=[r for r in selected if r['tool']==tool]
            for flag in sorted({r['failure'] for r in rows}):
                count=sum(r['failure']==flag for r in rows)
                output['failures'].append({'tier':tier,'tool':tool,'failure':flag,
                    'count':count,'total_runs':len(rows),'proportion':count/len(rows)})
            if tier=='T1':
                for category in sorted({r['category'] for r in rows}):
                    s=summarize_records([r for r in rows if r['category']==category],'IoU',tier,tool)
                    output['category_iou'].append({'tool':tool,'category':category,'mean':s['mean'],
                        'ci95':bootstrap_ci(list(s['task_means'].values()),rng,n_boot)})
    return output
