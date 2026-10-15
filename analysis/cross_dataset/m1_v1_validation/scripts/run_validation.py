#!/usr/bin/env python3
"""Harmonised, animal-aware M1 -> V1 test of the activity-response gene set against rheobase.

Statistic: mean gene-wise partial rank correlation within type, with animal-cluster bootstrap
intervals, leave-one-animal/type/gene influence, normalisation and covariate sensitivity, and a
pooling benchmark. Writes tables/ and figures/ next to scripts/.

Usage:  uv run python analysis/cross_dataset/m1_v1_validation/scripts/run_validation.py
"""
from __future__ import annotations

import json, os, re, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get('NDD_VALIDATION_ROOT', Path(__file__).resolve().parents[4])).resolve()
OUT = Path(os.environ.get('NDD_VALIDATION_OUT', Path(__file__).resolve().parents[1])).resolve()
TAB, FIG = [OUT / x for x in ("tables", "figures")]
for p in (TAB, FIG): p.mkdir(parents=True, exist_ok=True)
SEED = 20260802
RNG = np.random.default_rng(SEED)
NBOOT = 2000
GENES = ['Atf3','Atf4','Atf5','Atf7','Btg1','Btg2','Creb1','Creb3','Creb5','Crem','Egr1','Egr2','Egr3','Egr4','Elk1','Elk3','Elk4','Fos','Fosb','Fosl1','Fosl2','Ier2','Ier3','Ier5','Jun','Junb','Jund','Klf10','Klf11','Klf2','Klf4','Klf6','Klf9','Npas4','Nr4a1','Nr4a2','Nr4a3','Srf']
HK = ['Gapdh','Actb','Actg1','Tuba1a','Tuba1b','Tubb5','Ubb','Ubc','Eef1a1','Eef2','Rplp0','Rps18','Rpl13a','Ppia','Pgk1','Ldha','Aldoa','Eno1','Atp5b','Hprt','Tbp','B2m','Sdha','Ywhaz']


def design(df, type_col, numeric, categorical=()):
    blocks=[np.ones((len(df),1))]
    for c in numeric:
        v=pd.to_numeric(df[c],errors='coerce')
        miss=v.isna().astype(float).to_numpy()
        v=v.fillna(v.median()).to_numpy(float)
        r=stats.rankdata(v)
        if np.std(r)>0: blocks.append(((r-r.mean())/r.std())[:,None])
        if miss.sum() and miss.sum()<len(miss): blocks.append(miss[:,None])
    dum=pd.get_dummies(df[type_col].astype(str),drop_first=True,dtype=float)
    if dum.shape[1]: blocks.append(dum.to_numpy())
    for c in categorical:
        dum=pd.get_dummies(df[c].astype(str),prefix=c,drop_first=True,dtype=float)
        if dum.shape[1]: blocks.append(dum.to_numpy())
    X=np.column_stack(blocks)
    # remove dependent/constant columns deterministically
    keep=[]; rank=0
    for j in range(X.shape[1]):
        nr=np.linalg.matrix_rank(X[:,keep+[j]])
        if nr>rank: keep.append(j); rank=nr
    return X[:,keep]

def residualize_rank(A,X):
    A=np.asarray(A,float)
    if A.ndim==1: A=A[:,None]
    R=np.apply_along_axis(stats.rankdata,0,A)
    return R-X@np.linalg.lstsq(X,R,rcond=None)[0]

def set_effect(df,E,genes,type_col,numeric,categorical=()):
    X=design(df,type_col,numeric,categorical)
    y=residualize_rank(df['rheobase'].to_numpy(),X)[:,0]
    G=residualize_rank(E[genes].to_numpy(),X)
    y=(y-y.mean())/y.std(ddof=1)
    G=(G-G.mean(0))/G.std(0,ddof=1)
    r=(G*y[:,None]).sum(0)/(len(df)-1)
    return float(np.nanmean(r)),pd.Series(r,index=genes),X.shape[1]

def cluster_boot(df,E,genes,type_col,numeric,categorical=(),nboot=NBOOT):
    animals=pd.unique(df.animal)
    vals=[]
    groups={a:np.flatnonzero(df.animal.to_numpy()==a) for a in animals}
    for _ in range(nboot):
        draw=RNG.choice(animals,len(animals),replace=True)
        ix=np.concatenate([groups[a] for a in draw])
        d=df.iloc[ix].reset_index(drop=True); e=E.iloc[ix].reset_index(drop=True)
        try: vals.append(set_effect(d,e,genes,type_col,numeric,categorical)[0])
        except Exception: vals.append(np.nan)
    v=np.asarray(vals,float); v=v[np.isfinite(v)]
    return v, np.quantile(v,[.025,.975]), 2*min((v<=0).mean(),(v>=0).mean())

def load_m1():
    meta=pd.read_csv(ROOT/'PatchSeqExpression/m1_patchseq_meta_data.csv',sep='\t',low_memory=False).set_index('Cell')
    ep=pd.read_csv(ROOT/'m1_patchseq_ephys_features.csv').set_index('cell id')
    ids=meta.index[meta['RNA type'].astype(str).str.match(r'^L5 (IT|ET)_\d+$',na=False)]
    ids=ids.intersection(ep.index)
    ids=ids[pd.to_numeric(ep.loc[ids,'Rheobase (pA)'],errors='coerce').notna()]
    path=ROOT/'PatchSeqExpression/GSE163764_m1_patchseq_exon_counts.csv'
    wanted=set(GENES+HK+['Nptx1','Nptx2','Bdnf']); lib=pd.Series(0.0,index=ids);det=pd.Series(0,index=ids);parts=[];ngenes=0
    for ch in pd.read_csv(path,index_col=0,usecols=lambda x: x=='Unnamed: 0' or x in set(ids),chunksize=2000):
        ch.index=ch.index.astype(str);ngenes+=len(ch);lib=lib.add(ch.sum(axis=0),fill_value=0);det=det.add((ch>0).sum(axis=0),fill_value=0);hit=ch.index.intersection(wanted)
        if len(hit):parts.append(ch.loc[hit])
    gxc=pd.concat(parts).groupby(level=0).sum().reindex(sorted(wanted),fill_value=0);c=gxc.T.reindex(ids).astype(np.float32)
    # Exact percentile rank of each set gene against the complete gene
    # vector, streamed to keep memory bounded. Midrank handles tied zeros.
    less=np.zeros((len(ids),len(GENES)));equal=np.zeros_like(less);thr=c.reindex(columns=GENES,fill_value=0).to_numpy()
    for ch in pd.read_csv(path,index_col=0,usecols=lambda x: x=='Unnamed: 0' or x in set(ids),chunksize=1000):
        a=ch.to_numpy().T[:,:,None];less+=(a<thr[:,None,:]).sum(1);equal+=(a==thr[:,None,:]).sum(1)
    rankpct=pd.DataFrame((less+(equal+1)/2)/ngenes,index=range(len(ids)),columns=GENES)
    d=meta.loc[ids].copy(); d['cell_id']=d.index; d['animal']=d['Mouse'].astype(str)
    d['type']=d['RNA type'].astype(str); d['rheobase']=ep.loc[ids,'Rheobase (pA)'].astype(float)
    d['log_libsize']=np.log10(lib.loc[ids].replace(0,np.nan)); d['n_detected']=det.loc[ids]
    d['batch']=d['Sequencing batch'].astype(str); d['hold']=pd.to_numeric(d['Hold Time (min)'],errors='coerce')
    d['age']=pd.to_numeric(d['Mouse age'],errors='coerce'); d['sex']=d['Mouse gender'].astype(str)
    logcpm=np.log1p(c.div(lib.loc[ids],axis=0)*1e6).astype(np.float32)
    return d.reset_index(drop=True),c.reset_index(drop=True),logcpm.reset_index(drop=True),rankpct

def load_v1():
    qdir=ROOT/'ReplicationData/MouseV1_Sorensen2026/quant'
    frames={}; ranks={}; libs={}; dets={}; top50s={}
    for p in sorted(qdir.glob('*.parquet')):
        z=pd.read_parquet(p,columns=['gene','NumReads']);s=z.set_index('gene').NumReads;frames[p.stem]=s.reindex(sorted(set(GENES+HK)),fill_value=0);libs[p.stem]=s.sum();dets[p.stem]=(s>0).sum();top50s[p.stem]=s.nlargest(50).sum()/max(s.sum(),1);r=stats.rankdata(s.to_numpy(),method='average')/len(s);ranks[p.stem]=pd.Series(r,index=s.index).reindex(GENES,fill_value=np.nan)
    cnt=pd.DataFrame(frames).T.fillna(0).astype(np.float32); cnt.index=cnt.index.astype(str);rankpct=pd.DataFrame(ranks).T
    lib=pd.Series(libs); det=pd.Series(dets)
    met=pd.read_csv(ROOT/'ReplicationData/MouseV1_Sorensen2026/inferred_met_types.csv').rename(columns={'Unnamed: 0':'cell_id'}); met.cell_id=met.cell_id.astype(str); met=met.set_index('cell_id')
    ep=pd.read_csv(ROOT/'ReplicationData/MouseV1_Sorensen2026/exc_mMET_ephys_features.csv'); ep.specimen_id=ep.specimen_id.astype(str);ep=ep.set_index('specimen_id')
    man=pd.read_csv(ROOT/'ReplicationData/MouseV1_Sorensen2026/exc_patchseq_file_manifest.csv',encoding='utf-8-sig'); man=man[man.technique.eq('electrophysiology')].copy();man.specimen_id=man.specimen_id.astype(str)
    man['animal']=man.file_name.str.extract(r'^sub-([^_]+)_')[0]; man=man.set_index('specimen_id')
    ids=cnt.index.intersection(met.index).intersection(ep.index).intersection(man.index)
    ids=ids[met.loc[ids,'inferred_met_type'].astype(str).str.startswith('L5 ET') & ep.loc[ids,'rheobase_i'].notna()]
    d=pd.DataFrame(index=ids);d['cell_id']=ids;d['animal']=man.loc[ids,'animal'];d['type']=met.loc[ids,'inferred_met_type'].astype(str);d['rheobase']=ep.loc[ids,'rheobase_i'].astype(float)
    d['log_libsize']=np.log10(lib.loc[ids].replace(0,np.nan));d['n_detected']=det.loc[ids]
    c=cnt.loc[ids]; cpm=c.div(lib.loc[ids],axis=0)*1e6;d['top50_share']=pd.Series(top50s).loc[ids]
    logcpm=np.log1p(cpm).astype(np.float32)
    return d.reset_index(drop=True),c.reset_index(drop=True),logcpm.reset_index(drop=True),rankpct.loc[ids].reset_index(drop=True)

def normalizations(counts,logcpm,rankpct,genes):
    out={'logCPM':logcpm[genes].copy()}
    h=[g for g in HK if g in logcpm.columns and (counts[g]>0).mean()>=.5]
    out['housekeeping_ratio']=logcpm[genes].sub(logcpm[h].mean(axis=1),axis=0)
    out['within_cell_rank']=rankpct[genes].copy()
    return out

def run_cohort(name,d,c,lc,numeric):
    detection=pd.DataFrame({'gene':GENES,'present':[g in c for g in GENES],'detection_fraction':[float((c[g]>0).mean()) if g in c else 0 for g in GENES]})
    detection['cohort']=name; detection.to_csv(TAB/f'{name}_gene_detection.csv',index=False)
    return detection

def influence(name,d,E,genes,numeric,categorical=()):
    base,per,k=set_effect(d,E,genes,'type',numeric,categorical)
    rows=[]
    for a in pd.unique(d.animal):
        ix=d.animal.ne(a); eff=set_effect(d.loc[ix].reset_index(drop=True),E.loc[ix].reset_index(drop=True),genes,'type',numeric,categorical)[0]
        rows.append({'cohort':name,'animal':a,'effect':eff,'sign_flip':np.sign(eff)!=np.sign(base),'n_removed':int((~ix).sum())})
    pd.DataFrame(rows).to_csv(TAB/f'{name}_leave_one_animal.csv',index=False)
    tr=[]
    for t in pd.unique(d.type):
        ix=d.type.ne(t);eff=set_effect(d.loc[ix].reset_index(drop=True),E.loc[ix].reset_index(drop=True),genes,'type',numeric,categorical)[0]
        tr.append({'cohort':name,'type':t,'effect':eff,'sign_flip':np.sign(eff)!=np.sign(base),'n_removed':int((~ix).sum())})
    pd.DataFrame(tr).to_csv(TAB/f'{name}_leave_one_type.csv',index=False)
    gr=[]
    for g in genes:
        gg=[x for x in genes if x!=g];eff=set_effect(d,E,gg,'type',numeric,categorical)[0]
        gr.append({'cohort':name,'gene':g,'effect':eff,'sign_flip':np.sign(eff)!=np.sign(base)})
    pd.DataFrame(gr).to_csv(TAB/f'{name}_leave_one_gene.csv',index=False)
    return base,per,k

def animal_equal_effect(df,E,genes,type_col,numeric,categorical=()):
    """Sensitivity giving every animal equal weight after cell-level adjustment.

    Ranked phenotype and expression are residualized with the primary fixed
    design at cell level. Residuals are then averaged within animal and the
    gene-wise correlations are calculated across animal means.
    """
    X=design(df,type_col,numeric,categorical)
    y=residualize_rank(df['rheobase'].to_numpy(),X)[:,0]
    G=residualize_rank(E[genes].to_numpy(),X)
    tmp=pd.DataFrame(G,columns=genes)
    tmp['rheobase_resid']=y; tmp['animal']=df.animal.to_numpy()
    a=tmp.groupby('animal',sort=False).mean(numeric_only=True)
    yy=a.pop('rheobase_resid').to_numpy()
    yy=(yy-yy.mean())/yy.std(ddof=1)
    GG=a[genes].to_numpy(); GG=(GG-GG.mean(0))/GG.std(0,ddof=1)
    r=(GG*yy[:,None]).sum(0)/(len(a)-1)
    return float(np.nanmean(r)),pd.Series(r,index=genes),len(a)

def animal_equal_boot(df,E,genes,type_col,numeric,categorical=(),nboot=NBOOT):
    animals=pd.unique(df.animal); groups={a:np.flatnonzero(df.animal.to_numpy()==a) for a in animals}; vals=[]
    for _ in range(nboot):
        draw=RNG.choice(animals,len(animals),replace=True); chunks=[]; echunks=[]
        for j,a in enumerate(draw):
            ix=groups[a]; q=df.iloc[ix].copy(); q['animal']=f'bootstrap_{j}'; chunks.append(q); echunks.append(E.iloc[ix])
        d=pd.concat(chunks,ignore_index=True); e=pd.concat(echunks,ignore_index=True)
        try: vals.append(animal_equal_effect(d,e,genes,type_col,numeric,categorical)[0])
        except Exception: vals.append(np.nan)
    v=np.asarray(vals,float); v=v[np.isfinite(v)]
    return v,np.quantile(v,[.025,.975]),2*min((v<=0).mean(),(v>=0).mean())

def pooling_benchmark(m1,counts_cell):
    meta=pd.read_csv(ROOT/'PatchSeqExpression/m1_patchseq_meta_data.csv',sep='\t',low_memory=False).set_index('Cell'); ep=pd.read_csv(ROOT/'m1_patchseq_ephys_features.csv').set_index('cell id')
    ids=pd.Index(m1.cell_id)
    feats=['Rheobase (pA)','Input resistance (MOhm)','AP threshold (mV)','AP width (ms)','Resting membrane potential (mV)','Membrane time constant (ms)','Max number of APs']; genes=['Nptx1','Nptx2','Jun','Bdnf']
    lc=pd.DataFrame(counts_cell[genes].to_numpy(),index=ids,columns=genes);lc=np.log1p(lc.div(10**m1.set_index('cell_id').loc[ids,'log_libsize'],axis=0)*1e6)
    rows=[]
    models=[('naive',[],None),('family',[],'RNA family'),('fine_type',[],'RNA type'),('type_technical',['log_libsize','n_detected','hold'],'RNA type')]
    base=meta.loc[ids].copy();base['log_libsize']=m1.set_index('cell_id').loc[ids,'log_libsize'];base['n_detected']=m1.set_index('cell_id').loc[ids,'n_detected'];base['hold']=pd.to_numeric(base['Hold Time (min)'],errors='coerce')
    for model,num,tc in models:
      pidx=[]
      for f in feats:
       ok=pd.to_numeric(ep.loc[ids,f],errors='coerce').notna();ii=ids[ok];dd=base.loc[ii].copy();dd['y']=ep.loc[ii,f].astype(float)
       X=np.ones((len(ii),1)) if tc is None else design(dd,tc,num)
       y=residualize_rank(dd.y.to_numpy(),X)[:,0]
       for g in genes:
        x=residualize_rank(lc.loc[ii,g].to_numpy(),X)[:,0];r=stats.pearsonr(x,y).statistic;p=stats.pearsonr(x,y).pvalue
        pidx.append({'model':model,'feature':f,'gene':g,'n':len(ii),'rho':r,'p':p})
      z=pd.DataFrame(pidx);z['q_bh']=multipletests(z.p,method='fdr_bh')[1];rows.append(z)
    R=pd.concat(rows);R.to_csv(TAB/'pooling_benchmark.csv',index=False)
    R.groupby('model').agg(tests=('p','size'),hits_q05=('q_bh',lambda x:int((x<.05).sum())),median_abs_rho=('rho',lambda x:float(np.median(np.abs(x))))).reset_index().to_csv(TAB/'pooling_benchmark_summary.csv',index=False)

def plots(primary,spec):
    plt.style.use('seaborn-v0_8-whitegrid')
    # animal structure
    maps=[]
    for n in ['m1','v1']:
        z=pd.read_csv(TAB/f'{n}_specimen_animal_mapping.csv');q=z.animal.value_counts();maps.append((n,q))
    fig,ax=plt.subplots(1,2,figsize=(9,3.5));
    for a,(n,q) in zip(ax,maps):a.hist(q,bins=np.arange(1,q.max()+2)-.5);a.set(title=n.upper(),xlabel='cells per animal',ylabel='animals')
    fig.tight_layout();fig.savefig(FIG/'01_cohort_animal_structure.png',dpi=180);plt.close(fig)
    h=pd.read_csv(TAB/'historical_vs_harmonized.csv')
    fig,ax=plt.subplots(figsize=(9,4));ax.bar(np.arange(len(h)),h.effect,color=['#999999']*4+['#4477aa','#cc6677']);ax.axhline(0,color='k',lw=.8);ax.set_xticks(np.arange(len(h)),h.analysis,rotation=30,ha='right');ax.set_ylabel('reported effect (different estimands)');fig.tight_layout();fig.savefig(FIG/'02_historical_vs_harmonized.png',dpi=180);plt.close(fig)
    # primary
    fig,ax=plt.subplots(figsize=(6,3.5));x=np.arange(len(primary));ax.errorbar(x,primary.effect,yerr=[primary.effect-primary.ci_lo,primary.ci_hi-primary.effect],fmt='o',capsize=5);ax.axhline(0,color='k',lw=.8);ax.set_xticks(x,primary.cohort);ax.set_ylabel('mean gene-wise partial rho');fig.tight_layout();fig.savefig(FIG/'03_primary_animal_aware.png',dpi=180);plt.close(fig)
    # influence combined
    fig,axs=plt.subplots(1,3,figsize=(12,3.5))
    for cohort,color in [('m1','#4477aa'),('v1','#cc6677')]:
      for j,suf in enumerate(['animal','type','gene']):
       z=pd.read_csv(TAB/f'{cohort}_leave_one_{suf}.csv');axs[j].plot(np.arange(len(z)),z.effect,'.',alpha=.65,label=cohort,color=color)
    for j,t in enumerate(['animal','type','gene']):axs[j].axhline(0,color='k',lw=.7);axs[j].set_title('leave-one-'+t+'-out')
    axs[0].legend();fig.tight_layout();fig.savefig(FIG/'04_05_influence.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));
    for i,(cohort,g) in enumerate(spec.groupby('cohort')):ax.plot(np.arange(len(g)),g.effect,'o-',label=cohort)
    ax.axhline(0,color='k',lw=.7);ax.set_xticks(np.arange(len(g)),g.normalization,rotation=30,ha='right');ax.set_ylabel('set effect');ax.legend();fig.tight_layout();fig.savefig(FIG/'06_normalization_specification.png',dpi=180);plt.close(fig)
    a=pd.read_csv(TAB/'m1_per_gene_effects.csv').set_index('gene');b=pd.read_csv(TAB/'v1_per_gene_effects.csv').set_index('gene');j=a.join(b,lsuffix='_m1',rsuffix='_v1')
    fig,ax=plt.subplots(figsize=(5,5));ax.scatter(j.effect_m1,j.effect_v1);ax.axhline(0,color='grey',lw=.7);ax.axvline(0,color='grey',lw=.7);ax.set(xlabel='M1 partial rho',ylabel='V1 partial rho');fig.tight_layout();fig.savefig(FIG/'07_per_gene_scatter.png',dpi=180);plt.close(fig)
    pb=pd.read_csv(TAB/'pooling_benchmark.csv');fig,ax=plt.subplots(figsize=(7,4));data=[pb.loc[pb.model==m,'rho'] for m in pb.model.unique()];ax.boxplot(data,tick_labels=pb.model.unique());ax.set_ylabel('partial rho');ax.tick_params(axis='x',rotation=25);fig.tight_layout();fig.savefig(FIG/'08_pooling_benchmark.png',dpi=180);plt.close(fig)

def main():
    t0=time.time();warnings.filterwarnings('ignore',category=RuntimeWarning)
    # Load cohorts sequentially: the source count matrices are large and there
    # is no scientific reason to retain both in memory simultaneously.
    m1,mc,mlc,mrank=load_m1(); assert m1.cell_id.is_unique and m1.animal.notna().all()
    mm=m1[['cell_id','animal','type','batch','hold','age','sex']].copy();mm['mapping_source']='PatchSeqExpression/m1_patchseq_meta_data.csv: Mouse';mm['mapping_status']='verified_explicit';mm.to_csv(TAB/'m1_specimen_animal_mapping.csv',index=False)
    dm=run_cohort('m1',m1,mc,mlc,['log_libsize','n_detected','hold','age'])
    del mc,mlc,mrank
    # Determine V1 exposure eligibility in a separate load, then use the exact
    # cross-cohort gene intersection.
    v1,vc,vlc,vrank=load_v1(); assert v1.cell_id.is_unique and v1.animal.notna().all()
    vv=v1[['cell_id','animal','type']].copy();vv['mapping_source']='official exc_patchseq_file_manifest.csv: sub-{subject}';vv['mapping_status']='verified_documented_filename';vv.to_csv(TAB/'v1_specimen_animal_mapping.csv',index=False)
    flow=pd.DataFrame([['M1','L5 excitatory + finite rheobase + expression + animal',len(m1)],['V1','local quant + L5 ET + finite rheobase + manifest animal',len(v1)]],columns=['cohort','stage','n']);flow.to_csv(TAB/'cohort_flow.csv',index=False)
    dv=run_cohort('v1',v1,vc,vlc,['log_libsize','n_detected','top50_share'])
    gd=dm[['gene','present','detection_fraction']].merge(dv[['gene','present','detection_fraction']],on='gene',suffixes=('_m1','_v1'));gd['primary_included']=gd.present_m1&gd.present_v1&(gd.detection_fraction_m1>=.25)&(gd.detection_fraction_v1>=.25);gd['source']='registered IEG_TF';gd.to_csv(TAB/'gene_set_definition.csv',index=False);genes=gd.loc[gd.primary_included,'gene'].tolist()
    specs=[];primary=[]
    # Release the V1 exposure matrix used only to establish the common panel;
    # analyze M1, release M1 matrices, then reload and analyze V1.
    del v1,vc,vlc,vrank
    m1,mc,mlc,mrank=load_m1()
    configs=[('m1',m1,mc,mlc,mrank,['log_libsize','n_detected','hold','age'],['batch','sex'])]
    for name,d,c,lc,rp,num,cat in configs:
      norms=normalizations(c,lc,rp,genes)
      for norm,E in norms.items():
       eff,per,k=set_effect(d,E,genes,'type',num,cat)
       boot,ci,p=cluster_boot(d,E,genes,'type',num,cat)
       specs.append({'cohort':name,'normalization':norm,'covariates':'primary','effect':eff,'ci_lo':ci[0],'ci_hi':ci[1],'bootstrap_sign_p':p,'n':len(d),'animals':d.animal.nunique(),'genes':len(genes)})
       pd.DataFrame({'replicate':range(len(boot)),'effect':boot}).to_csv(TAB/f'{name}_{norm}_bootstrap_results.csv',index=False)
       if norm=='logCPM':
        pd.DataFrame({'replicate':range(len(boot)),'effect':boot}).to_csv(TAB/f'{name}_bootstrap_results.csv',index=False)
        base,per,k=influence(name,d,E,genes,num,cat);pd.DataFrame({'gene':per.index,'effect':per.values}).to_csv(TAB/f'{name}_per_gene_effects.csv',index=False)
        primary.append({'cohort':name,'cells':len(d),'animals':d.animal.nunique(),'types':d.type.nunique(),'genes':len(genes),'effect':base,'ci_lo':ci[0],'ci_hi':ci[1],'bootstrap_sign_p':p,'design_columns':k})
      # covariate curve primary logCPM
      covsets=[('type_only',[]),('depth_detection',['log_libsize','n_detected']),('without_composition',[x for x in num if x not in ['top50_share','hold']])]
      for lab,nn in covsets:
       cats=cat if lab in ('primary','without_composition') else []
       eff,_,_=set_effect(d,norms['logCPM'],genes,'type',nn,cats);specs.append({'cohort':name,'normalization':'logCPM','covariates':lab,'effect':eff,'ci_lo':np.nan,'ci_hi':np.nan,'bootstrap_sign_p':np.nan,'n':len(d),'animals':d.animal.nunique(),'genes':len(genes)})
    # Save M1 population sensitivity while its matrix is resident.
    ix=m1.type.str.startswith('L5 ET');eff,_,_=set_effect(m1.loc[ix].reset_index(drop=True),mlc.loc[ix].reset_index(drop=True),genes,'type',['log_libsize','n_detected','hold','age'],['batch','sex']);pd.DataFrame([{'population':'M1 L5 ET','n':ix.sum(),'animals':m1.loc[ix,'animal'].nunique(),'effect':eff}]).to_csv(TAB/'m1_population_sensitivity.csv',index=False)
    m1_counts_for_pool=mc[ ['Nptx1','Nptx2','Jun','Bdnf'] ].copy()
    del mc,mlc,mrank,norms
    v1,vc,vlc,vrank=load_v1()
    name,d,c,lc,num='v1',v1,vc,vlc,['log_libsize','n_detected','top50_share']
    norms=normalizations(c,lc,vrank,genes)
    for norm,E in norms.items():
       eff,per,k=set_effect(d,E,genes,'type',num)
       boot,ci,p=cluster_boot(d,E,genes,'type',num)
       specs.append({'cohort':name,'normalization':norm,'covariates':'primary','effect':eff,'ci_lo':ci[0],'ci_hi':ci[1],'bootstrap_sign_p':p,'n':len(d),'animals':d.animal.nunique(),'genes':len(genes)})
       pd.DataFrame({'replicate':range(len(boot)),'effect':boot}).to_csv(TAB/f'{name}_{norm}_bootstrap_results.csv',index=False)
       if norm=='logCPM':
        pd.DataFrame({'replicate':range(len(boot)),'effect':boot}).to_csv(TAB/f'{name}_bootstrap_results.csv',index=False)
        base,per,k=influence(name,d,E,genes,num);pd.DataFrame({'gene':per.index,'effect':per.values}).to_csv(TAB/f'{name}_per_gene_effects.csv',index=False)
        primary.append({'cohort':name,'cells':len(d),'animals':d.animal.nunique(),'types':d.type.nunique(),'genes':len(genes),'effect':base,'ci_lo':ci[0],'ci_hi':ci[1],'bootstrap_sign_p':p,'design_columns':k})
    covsets=[('type_only',[]),('depth_detection',['log_libsize','n_detected']),('without_composition',[x for x in num if x not in ['top50_share','hold']])]
    for lab,nn in covsets:
       eff,_,_=set_effect(d,norms['logCPM'],genes,'type',nn);specs.append({'cohort':name,'normalization':'logCPM','covariates':lab,'effect':eff,'ci_lo':np.nan,'ci_hi':np.nan,'bootstrap_sign_p':np.nan,'n':len(d),'animals':d.animal.nunique(),'genes':len(genes)})
    # Equal-animal-weight sensitivity: average adjusted residuals within animal.
    m1_reload,_,m1_lc,_=load_m1()
    ae_m1,_,_=animal_equal_effect(m1_reload,m1_lc,genes,'type',['log_libsize','n_detected','hold','age'],['batch','sex'])
    ae_v1,_,_=animal_equal_effect(v1,vlc,genes,'type',['log_libsize','n_detected','top50_share'])
    aeb_m1,aeci_m1,aep_m1=animal_equal_boot(m1_reload,m1_lc,genes,'type',['log_libsize','n_detected','hold','age'],['batch','sex'])
    aeb_v1,aeci_v1,aep_v1=animal_equal_boot(v1,vlc,genes,'type',['log_libsize','n_detected','top50_share'])
    pd.DataFrame({'replicate':range(len(aeb_m1)),'effect':aeb_m1}).to_csv(TAB/'m1_animal_equal_bootstrap_results.csv',index=False)
    pd.DataFrame({'replicate':range(len(aeb_v1)),'effect':aeb_v1}).to_csv(TAB/'v1_animal_equal_bootstrap_results.csv',index=False)
    pd.DataFrame([
        {'cohort':'m1','cells':len(m1_reload),'animals':m1_reload.animal.nunique(),'effect':ae_m1,'ci_lo':aeci_m1[0],'ci_hi':aeci_m1[1],'bootstrap_sign_p':aep_m1,'estimand':'correlation across animal-mean adjusted residuals'},
        {'cohort':'v1','cells':len(v1),'animals':v1.animal.nunique(),'effect':ae_v1,'ci_lo':aeci_v1[0],'ci_hi':aeci_v1[1],'bootstrap_sign_p':aep_v1,'estimand':'correlation across animal-mean adjusted residuals'}
    ]).to_csv(TAB/'animal_equal_weight_sensitivity.csv',index=False)
    P=pd.DataFrame(primary);P.to_csv(TAB/'primary_cohort_results.csv',index=False);S=pd.DataFrame(specs).drop_duplicates();S.to_csv(TAB/'normalization_and_covariate_specifications.csv',index=False)
    hist=pd.DataFrame([['V1 gene-wise set test','mean gene-wise partial rho',-.0455540831],['V1 exploratory composite','composite index rho',-.1215008552],['M1 original partial score','composite score rho',-.2493031738],['M1 housekeeping + composition','composite index rho',-.1448654387],['M1 harmonized','mean gene-wise partial rho',P.loc[P.cohort.eq('m1'),'effect'].iloc[0]],['V1 harmonized','mean gene-wise partial rho',P.loc[P.cohort.eq('v1'),'effect'].iloc[0]]],columns=['analysis','estimand','effect']);hist.to_csv(TAB/'historical_vs_harmonized.csv',index=False)
    # Release V1 matrices before the focused pooling benchmark reloads M1.
    del vc,vlc,vrank,norms,c,lc
    pooling_benchmark(m1,m1_counts_for_pool);plots(P,S[S.covariates.eq('primary')])
    print(json.dumps({'primary':primary,'runtime_seconds':time.time()-t0},indent=2))

if __name__=='__main__': main()
