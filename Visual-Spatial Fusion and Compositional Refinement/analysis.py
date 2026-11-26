import math
import json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from statistical_core import row_ranks as rankdata, friedman_test, holm, rank_biserial

def spearman_value(a,b):
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])

ROOT = Path(__file__).resolve().parent

def load(name):
    return json.loads((ROOT / 'data' / f'SIMULATED_{name}.json').read_text(encoding='utf-8'))

def bootstrap(values, seed=42, n_boot=10000):
    x = np.asarray(values, dtype=float)

    rng = np.random.default_rng(seed)
    draws = x[rng.integers(len(x), size=(n_boot, len(x)))].mean(axis=1)
    return [float(x.mean()), *np.quantile(draws, [.025, .975]).tolist()]

def signed_rank(d):

    d = np.round(np.asarray(d, float), 12)
    d = d[d != 0]
    if not len(d):
        return 1.0, 0.0
    ranks = np.rint(2 * rankdata(abs(d))).astype(int)
    positive = int(ranks[d > 0].sum())
    total = int(ranks.sum())
    counts = [0] * (total + 1)
    counts[0] = 1
    for rank in ranks:
        for k in range(total, int(rank)-1, -1):
            counts[k] += counts[k-int(rank)]
    p = min(1., 2 * sum(counts[:min(positive, total-positive)+1]) / 2**len(d))
    return p, float((2*positive-total)/total)

def paired_analysis(table):

    a = np.round(table.to_numpy(float), 12)
    if np.all(a == a[:, :1]):
        omnibus = {'statistic': 0., 'p': 1.}
    else:
        omnibus = friedman_test(a)
    pairs = []
    rng = np.random.default_rng(42)
    for left, right in combinations(table.columns, 2):
        d = (table[left]-table[right]).to_numpy()
        p, effect = signed_rank(d)
        effects = rank_biserial(d[rng.integers(len(d), size=(10000,len(d)))])
        pairs.append({'a': left, 'b': right, 'p': p, 'difference': bootstrap(d),
                      'rank_biserial': effect, 'effect_ci': np.quantile(effects,[.025,.975]).tolist()})
    corrected = holm([p['p'] for p in pairs])
    for item, p in zip(pairs, corrected):
        item['p_holm'] = float(p)
    return {'n': len(table), 'friedman': omnibus, 'pairs': pairs}

def task_table(records, value, conditions, tier=None, budget=None, label='condition'):
    df = pd.DataFrame(records)
    if tier is not None:
        df = df[df['tier'] == tier]
    if budget is not None:
        df = df[df['budget'] == budget]

    table = df.groupby(['task',label])[value].mean().unstack().reindex(columns=conditions)

    return table

def analyze_all():
    report = {'synthetic': True, 'summaries': {}, 'statistics': {}}
    names_a = ['Naive','General edit','Constraint-explicit','Staged']
    names_b = ['A0 text','A1 assets','A2 structure','A3 both']
    names_c = ['Ref. assets + ref. structure','Pred. assets + ref. structure',
               'Ref. assets + pred. structure','Pred. assets + pred. structure']
    names_d = ['Single pass','Best-of-N','Prompt repair','Full workflow']
    A,B,C,D = [load(x+'_runs') for x in 'ABCD']
    for tier in ['T1','T2']:
        for metric in ['DINO','BG MAE','Seam'] + (['Center error','Scale error','Relations','FG MSE','Style error'] if tier=='T1' else []):
            table = task_table(A,metric,names_a,tier)
            report['summaries'][f'A {tier} {metric}'] = {c:bootstrap(table[c]) for c in table}
            if tier == 'T1' and metric in ['Scale error','FG MSE','Style error']:
                report['summaries'][f'A reduction {metric}'] = {
                    c:bootstrap((table['Naive']-table[c])/table['Naive']) for c in names_a[1:]}
            if tier=='T1' and metric=='Relations':
                report['statistics']['A T1 Relations'] = paired_analysis(table)
        table = task_table(B,'S',names_b,tier)
        report['summaries'][f'B {tier}'] = {c:bootstrap(table[c]) for c in table}
        report['summaries'][f'B {tier} interaction'] = bootstrap(table[names_b[3]]-table[names_b[1]]-table[names_b[2]]+table[names_b[0]])
        report['statistics'][f'B {tier}'] = paired_analysis(table)
        for budget in [4,8,16]:
            table = task_table(D,'joint_success',names_d,tier,budget,'method')
            report['summaries'][f'D {tier} {budget}'] = {c:bootstrap(table[c]) for c in table}
            report['statistics'][f'D {tier} {budget}'] = paired_analysis(table)
    table = task_table(C,'S',names_c)
    report['summaries']['C'] = {c:bootstrap(table[c]) for c in table}
    report['summaries']['C drops'] = {c:bootstrap(table[names_c[0]]-table[c]) for c in names_c[1:]}
    report['statistics']['C'] = paired_analysis(table)
    report['efficiency'] = {}
    df = pd.DataFrame(D)
    for (tier,method),group in df.groupby(['tier','method']):
        rates = group.groupby(['budget','task']).joint_success.mean().groupby('budget').mean()
        final = group[group.budget==16]
        successes = final.joint_success.sum()
        report['efficiency'][f'{tier} {method}'] = {
            'first_tested_budget': {str(t):next((int(b) for b,v in rates.items() if v>=t),None) for t in [.5,.8]},
            'assumed_cost_per_success_16':float(final.assumed_usd.sum()/successes) if successes else None}
    report['calibration'] = {name:{'rho':float(spearman_value(d['truth'],d['proxy'])),
        'eligible':bool(spearman_value(d['truth'],d['proxy'])>=.7)} for name,d in load('proxy_calibration').items()}
    return report

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