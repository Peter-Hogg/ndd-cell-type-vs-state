#!/usr/bin/env python3
"""Exploratory bridge from Patch-seq gene/ephys effects to RNA-seq spread.

Genes are ranked in a deterministic animal-disjoint M1 discovery split for seven
ephys endpoints. Their within-cluster spread is then measured in the independent
Allen Smart-seq atlas and compared with outcome-blind matched genes. Full-cohort
and held-out M1 effects, and V1 rheobase effects where available, are
descriptive. This workflow is hypothesis-generating.

Usage:  uv run python analysis/cross_dataset/ephys_spread/scripts/run_ephys_spread.py
"""
from __future__ import annotations
import hashlib, importlib.util, os, sys, time
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parents[1]
TAB=OUT/'tables'; FIG=OUT/'figures'
for p in (TAB,FIG): p.mkdir(parents=True,exist_ok=True)
VAL=ROOT/'analysis/cross_dataset/m1_v1_validation'
spec=importlib.util.spec_from_file_location('validation',VAL/'scripts/run_validation.py')
rv=importlib.util.module_from_spec(spec); spec.loader.exec_module(rv)
SEED=20260804; RNG=np.random.default_rng(SEED); TOP_K=30; MATCH_K=10
EPHYS=['Input resistance (MOhm)','AP threshold (mV)','AP width (ms)',
       'Resting membrane potential (mV)','ISI adaptation index',
       'Spike frequency adaptation','Rheobase (pA)']

def hash_split(a):
    u=hashlib.sha256(f'{SEED}:{a}'.encode()).digest()[0]
    return 'discovery' if u < 179 else 'heldout'  # approximately 70/30 by animal

def load_m1_full():
    d,_,_,_=rv.load_m1(); ids=pd.Index(d.cell_id)
    path=ROOT/'PatchSeqExpression/GSE163764_m1_patchseq_exon_counts.csv'
    chunks=[]; lib=pd.Series(0.,index=ids)
    for ch in pd.read_csv(path,index_col=0,usecols=lambda x:x=='Unnamed: 0' or x in set(ids),chunksize=2000):
        ch.index=ch.index.astype(str); lib=lib.add(ch.sum(axis=0),fill_value=0); chunks.append(ch.astype('float32'))
    c=pd.concat(chunks).groupby(level=0).sum(); c=c.loc[~c.index.duplicated()].T.reindex(ids)
    log=np.log1p(c.div(lib.loc[ids],axis=0)*1e6).astype('float32').reset_index(drop=True)
    gene=pd.Index(log.columns.astype(str)); keep=~gene.str.startswith('ERCC-') & ~gene.str.match(r'^(Gm\d|.*Rik$|LOC\d|A[A-Z]\d{6}|BC\d{6})') & (c.reset_index(drop=True)>0).mean(axis=0).ge(.25).to_numpy()
    return d,log.loc[:,keep],c.reset_index(drop=True).loc[:,keep]

def effects(d,E,y,mask):
    dd=d.loc[mask].reset_index(drop=True); EE=E.loc[mask].reset_index(drop=True); yy=pd.to_numeric(y.loc[mask],errors='coerce')
    ok=yy.notna().to_numpy(); dd=dd.loc[ok].reset_index(drop=True); EE=EE.loc[ok].reset_index(drop=True); dd['outcome']=yy.loc[ok].to_numpy()
    X=rv.design(dd,'type',['log_libsize','n_detected','hold','age'],['batch','sex'])
    yr=rv.residualize_rank(dd.outcome.to_numpy(),X)[:,0]; yr=(yr-yr.mean())/yr.std(ddof=1)
    vals=[]; genes=EE.columns.tolist()
    for i in range(0,len(genes),1200):
        g=genes[i:i+1200]; R=rv.residualize_rank(EE[g].to_numpy(),X); sd=R.std(axis=0,ddof=1); R=(R-R.mean(axis=0))/np.where(sd>0,sd,np.nan)
        vals.extend((R*yr[:,None]).sum(axis=0)/(len(dd)-1))
    return pd.Series(vals,index=genes),len(dd),dd.animal.nunique()

def m1_screen():
    d,E,C=load_m1_full(); ep=pd.read_csv(ROOT/'m1_patchseq_ephys_features.csv').set_index('cell id').reindex(d.cell_id)
    d['split']=d.animal.map(hash_split); rows=[]; selected=[]
    for endpoint in EPHYS:
        y=pd.Series(pd.to_numeric(ep[endpoint],errors='coerce').to_numpy(),index=d.index)
        ed,nd,ad=effects(d,E,y,d.split.eq('discovery')); eh,nh,ah=effects(d,E,y,d.split.eq('heldout')); ef,nf,af=effects(d,E,y,pd.Series(True,index=d.index))
        rank=ed.abs().rank(method='min',ascending=False).astype(int)
        q=pd.DataFrame({'gene':E.columns,'endpoint':endpoint,'rho_discovery':ed.values,'abs_rank_discovery':rank.values,'rho_heldout':eh.reindex(E.columns).values,'rho_full':ef.reindex(E.columns).values,'n_discovery':nd,'animals_discovery':ad,'n_heldout':nh,'animals_heldout':ah,'n_full':nf,'animals_full':af})
        q['selected_top30']=q.abs_rank_discovery.le(TOP_K); rows.append(q); selected.extend(q.loc[q.selected_top30,'gene'])
    R=pd.concat(rows,ignore_index=True); R.to_csv(TAB/'m1_gene_ephys_screen.csv',index=False)
    sel=sorted(set(selected)); pd.DataFrame({'gene':sel}).to_csv(TAB/'selected_gene_union.csv',index=False)
    metrics=pd.DataFrame({'gene':E.columns,'m1_detection':(C>0).mean(axis=0).values,'m1_mean_logcpm':E.mean(axis=0).values})
    return d,E,R,sel,metrics

def atlas_metrics(selected, m1metrics):
    h5=ROOT/'Gene_Expression/Data/expression_matrix.hdf5'; meta=pd.read_csv(ROOT/'Gene_Expression/Data/metadata.csv',low_memory=False).set_index('sample_name')
    with h5py.File(h5,'r') as f:
        genes=np.array([x.decode() for x in f['gene_names'][:]]); cells=np.array([x.decode() for x in f['sample_names'][:]])
        gi={g:i for i,g in enumerate(genes)}; lib=np.asarray(f['data/total_exon_counts'][:],float); grp=f['data/exon']; ptr=np.asarray(grp['p'][:],int); nall=int(grp['dims'][0])
        md=meta.reindex(cells); glut=md.class_label.astype(str).str.contains('Glutamatergic',na=False).to_numpy(); clu=md.cluster_label.astype(str).to_numpy(); vc=pd.Series(clu[glut]).value_counts(); keepcl=set(vc[vc>=20].index); use=glut & np.array([x in keepcl for x in clu]); ix=np.flatnonzero(use)
        cats=pd.Categorical(clu[ix]); code=cats.codes; ng=len(cats.categories); counts=np.bincount(code,minlength=ng)
        eligible=[g for g in m1metrics.gene if g in gi and g not in selected]
        pool=list(RNG.choice(eligible,size=min(6000,len(eligible)),replace=False)); wanted=sorted((set(selected)|set(pool))&set(gi))
        out=[]
        for g in wanted:
            j=gi[g]; lo,hi=ptr[j],ptr[j+1]; v=np.zeros(nall,dtype=float)
            if hi>lo: v[np.asarray(grp['i'][lo:hi],int)]=np.asarray(grp['x'][lo:hi],float)
            z=np.log1p(v[ix]/lib[ix]*1e6); means=np.bincount(code,weights=z,minlength=ng)/counts; resid=z-means[code]
            den=((z-z.mean())**2).sum(); eta=float((counts*(means-z.mean())**2).sum()/den) if den>0 else np.nan
            out.append({'gene':g,'atlas_detection':float((v[ix]>0).mean()),'atlas_mean_logcpm':float(z.mean()),'atlas_sd_within_cluster':float(resid.std(ddof=1)),'atlas_eta2_cluster':eta,'atlas_cells':len(ix),'atlas_clusters':ng})
    A=pd.DataFrame(out).merge(m1metrics,on='gene',how='left'); A.to_csv(TAB/'atlas_gene_spread_metrics.csv',index=False)
    return A

def matched_analysis(screen, selected, A):
    avail=A.dropna().set_index('gene'); features=['m1_detection','m1_mean_logcpm','atlas_detection','atlas_mean_logcpm']; Z=(avail[features]-avail[features].mean())/avail[features].std().replace(0,np.nan)
    controls=[g for g in avail.index if g not in selected]; matchrows=[]
    for g in selected:
        if g not in avail.index: continue
        near=((Z.loc[controls]-Z.loc[g])**2).sum(axis=1).nsmallest(MATCH_K).index
        vals=avail.loc[near,'atlas_sd_within_cluster']; obs=avail.loc[g,'atlas_sd_within_cluster']
        pct=(float((vals<obs).sum())+.5*float((vals==obs).sum()))/len(vals)
        z=(obs-vals.mean())/vals.std(ddof=1) if vals.std(ddof=1)>0 else np.nan
        for rank,c in enumerate(near,1): matchrows.append({'selected_gene':g,'control_gene':c,'match_rank':rank,'selected_spread':obs,'control_spread':avail.loc[c,'atlas_sd_within_cluster'],'selected_eta2':avail.loc[g,'atlas_eta2_cluster'],'control_eta2':avail.loc[c,'atlas_eta2_cluster'],'spread_percentile':pct,'spread_z_local':z})
    M=pd.DataFrame(matchrows); M.to_csv(TAB/'expression_matched_spread_controls.csv',index=False)
    one=M.drop_duplicates('selected_gene').set_index('selected_gene'); summaries=[]
    for endpoint,q in screen[screen.selected_top30].groupby('endpoint'):
        genes=[g for g in q.gene if g in one.index]; x=one.loc[genes,'spread_percentile']-.5
        w=stats.wilcoxon(x,alternative='two-sided') if len(x)>=5 and np.any(x!=0) else (np.nan,np.nan)
        concord=int((np.sign(q.rho_discovery)==np.sign(q.rho_heldout)).sum())
        summaries.append({'endpoint':endpoint,'selected_genes':len(q),'atlas_mapped':len(genes),'median_spread_percentile':float((x+.5).median()),'mean_spread_z_local':float(one.loc[genes,'spread_z_local'].mean()),'median_eta2_cluster':float(one.loc[genes,'selected_eta2'].median()),'wilcoxon_p_vs_half':float(w.pvalue) if hasattr(w,'pvalue') else np.nan,'median_abs_rho_discovery':float(q.rho_discovery.abs().median()),'median_abs_rho_heldout':float(q.rho_heldout.abs().median()),'heldout_sign_concordant':concord,'sign_concordance_heldout':concord/len(q),'binomial_p_sign_concordance':stats.binomtest(concord,len(q),.5).pvalue})
    S=pd.DataFrame(summaries); S['q_bh_spread']=multipletests(S.wilcoxon_p_vs_half.fillna(1),method='fdr_bh')[1]; S['q_bh_sign_concordance']=multipletests(S.binomial_p_sign_concordance,method='fdr_bh')[1]; S.to_csv(TAB/'endpoint_spread_summary.csv',index=False)
    return M,S

def v1_rheobase(selected):
    d,_,_,_=rv.load_v1(); qdir=ROOT/'ReplicationData/MouseV1_Sorensen2026/quant'; frames={}; libs={}
    for p in sorted(qdir.glob('*.parquet')):
        if p.stem not in set(d.cell_id): continue
        z=pd.read_parquet(p,columns=['gene','NumReads']); s=z.set_index('gene').NumReads; frames[p.stem]=s.reindex(selected,fill_value=0); libs[p.stem]=s.sum()
    C=pd.DataFrame(frames).T.reindex(d.cell_id); L=pd.Series(libs).reindex(d.cell_id); E=np.log1p(C.div(L,axis=0)*1e6).reset_index(drop=True)
    eff=rv.set_effect(d,E,[g for g in selected if g in E.columns],'type',['log_libsize','n_detected','top50_share'])[1]
    out=pd.DataFrame({'gene':eff.index,'v1_rheobase_rho':eff.values}); out.to_csv(TAB/'v1_selected_gene_rheobase_effects.csv',index=False); return out

def triage(screen,M,S,V):
    one=M.drop_duplicates('selected_gene')[['selected_gene','spread_percentile','spread_z_local','selected_eta2']]
    T=screen[screen.selected_top30].merge(one,left_on='gene',right_on='selected_gene',how='left').drop(columns='selected_gene').merge(V,on='gene',how='left')
    T['m1_sign_concordant']=np.sign(T.rho_discovery)==np.sign(T.rho_heldout)
    T['v1_sign_concordant_for_rheobase']=np.where(T.endpoint.eq('Rheobase (pA)'),np.sign(T.rho_discovery)==np.sign(T.v1_rheobase_rho),np.nan)
    T['heuristic_convergence']=T.m1_sign_concordant&T.rho_heldout.abs().ge(.2)&T.spread_percentile.ge(.8)
    T.to_csv(TAB/'candidate_triage.csv',index=False)

def figures(screen,S,M):
    fig,axs=plt.subplots(1,3,figsize=(16,5.2),constrained_layout=True)
    order=S.sort_values('median_spread_percentile'); axs[0].barh(order.endpoint,order.median_spread_percentile,color='#0072B2'); axs[0].axvline(.5,color='#555'); axs[0].set_xlabel('Median matched spread percentile'); axs[0].set_title('A  Atlas within-cluster spread')
    axs[1].scatter(S.median_abs_rho_discovery,S.median_abs_rho_heldout,c=S.median_spread_percentile,cmap='viridis',s=70); axs[1].axline((0,0),slope=1,color='#aaa',lw=1); axs[1].set(xlabel='Median |rho|, discovery',ylabel='Median |rho|, held-out',title='B  Winner’s-curse check')
    labels={'AP threshold (mV)':'AP threshold','AP width (ms)':'AP width','ISI adaptation index':'ISI adaptation','Input resistance (MOhm)':'Input resistance','Resting membrane potential (mV)':'Resting potential','Rheobase (pA)':'Rheobase','Spike frequency adaptation':'Spike adaptation'}
    offsets=[(5,5),(5,-12),(5,-12),(5,6),(5,5),(5,5),(5,-12)]
    for (_,r),off in zip(S.iterrows(),offsets): axs[1].annotate(labels[r.endpoint],(r.median_abs_rho_discovery,r.median_abs_rho_heldout),xytext=off,textcoords='offset points',fontsize=7)
    one=M.drop_duplicates('selected_gene'); axs[2].scatter(one.selected_eta2,one.spread_percentile,alpha=.6,color='#D55E00'); axs[2].axhline(.5,color='#555'); axs[2].set(xlabel='Atlas variance explained by cluster (eta²)',ylabel='Matched spread percentile',title='C  Cluster identity vs within-cluster spread')
    fig.savefig(FIG/'ephys_gene_atlas_spread.svg',metadata={'Date':None}); fig.savefig(FIG/'ephys_gene_atlas_spread.png',dpi=220); plt.close(fig)

def main():
    t=time.time(); d,E,screen,selected,metrics=m1_screen(); A=atlas_metrics(selected,metrics); M,S=matched_analysis(screen,selected,A); V=v1_rheobase(selected); triage(screen,M,S,V); figures(screen,S,M)
    print(S.to_string(index=False)); print(f'\nselected union={len(selected)} runtime={time.time()-t:.1f}s')

if __name__=='__main__': main()
