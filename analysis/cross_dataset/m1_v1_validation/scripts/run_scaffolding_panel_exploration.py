#!/usr/bin/env python3
"""Scaffolding and Ca/cAMP-CaMK panel set effects on rheobase in the M1 and V1 cohorts.

The panels are copied from `analysis/m1_patchseq/scripts/23_functional_categories_by_layer.py`.
The primary output is the mean gene-wise, type/technical-adjusted partial-rank
statistic from run_validation.py. Animal-cluster intervals quantify sampling
uncertainty. A secondary matched-random-set null compares each panel with
equal-sized sets matched on M1 and V1 abundance and detection; it is a
specificity diagnostic.

Usage:  uv run python analysis/cross_dataset/m1_v1_validation/scripts/run_scaffolding_panel_exploration.py
"""
from __future__ import annotations
import importlib.util, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[4]
CODE=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('NDD_VALIDATION_OUT',CODE)).resolve(); TAB=OUT/'tables'; TAB.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('validation',CODE/'scripts/run_validation.py')
rv=importlib.util.module_from_spec(spec); spec.loader.exec_module(rv)
SEED=20260803; RNG=np.random.default_rng(SEED); rv.RNG=np.random.default_rng(SEED)

PANELS={
 'ca_camp_adaptors':['Rcan1','Rcan2','Rcan3','Caln1','Calm1','Calm2','Calm3','Camkk1','Camkk2','Akap5','Akap6','Akap11','Akap12','Akap13'],
 'structural_late':['Baiap2','Baiap3','Sipa1l1','Sipa1l2','Synpo','Synpo2','Syngap1','Dlg4','Dlg3','Shank1','Shank2','Shank3'],
 'synaptic_plasticity':['Arc','Homer1','Homer1a','Bdnf','Nptx2','Nptx1','Nptxr','Syt4','Camk2g','Camk1g','Camk4','Mapk8ip2','Mapk1ip1l','Rgs2','Rgs4','Rgs14','Dusp1','Dusp4','Dusp5','Dusp6','Ppp1r1b','Ppp1r15a','Plk2','Plk3','Prkcz','Prkci','Rapgef4','Rapgef5','Rapgef3']}
PANELS['combined_scaffolding_camk']=PANELS['ca_camp_adaptors']+PANELS['structural_late']

def cohort_data():
    m1,_,_,_=rv.load_m1(); v1,_,_,_=rv.load_v1()
    mpath=ROOT/'PatchSeqExpression/GSE163764_m1_patchseq_exon_counts.csv'
    mgenes=pd.read_csv(mpath,usecols=[0]).iloc[:,0].astype(str)
    first=next(iter(sorted((ROOT/'ReplicationData/MouseV1_Sorensen2026/quant').glob('*.parquet'))))
    vgenes=pd.read_parquet(first,columns=['gene']).gene.astype(str)
    genes=sorted(set(mgenes)&set(vgenes))
    ids=pd.Index(m1.cell_id); chunks=[]; mlib=pd.Series(0.,index=ids)
    for ch in pd.read_csv(mpath,index_col=0,usecols=lambda x:x=='Unnamed: 0' or x in set(ids),chunksize=2000):
        ch.index=ch.index.astype(str); mlib=mlib.add(ch.sum(axis=0),fill_value=0); hit=ch.index.intersection(genes)
        if len(hit): chunks.append(ch.loc[hit].astype('float32'))
    mc=pd.concat(chunks).groupby(level=0).sum().reindex(genes,fill_value=0).T.reindex(ids)
    ml=np.log1p(mc.div(mlib.loc[ids],axis=0)*1e6).reset_index(drop=True)
    qdir=ROOT/'ReplicationData/MouseV1_Sorensen2026/quant'; wanted=set(v1.cell_id.astype(str)); frames={}; libs={}
    for p in sorted(qdir.glob('*.parquet')):
        if p.stem not in wanted: continue
        z=pd.read_parquet(p,columns=['gene','NumReads']); s=z.set_index('gene').NumReads
        frames[p.stem]=s.reindex(genes,fill_value=0).astype('float32'); libs[p.stem]=float(s.sum())
    vc=pd.DataFrame(frames).T.reindex(v1.cell_id.astype(str)); vlib=pd.Series(libs).reindex(vc.index)
    vl=np.log1p(vc.div(vlib,axis=0)*1e6).reset_index(drop=True)
    return m1,v1,mc.reset_index(drop=True),vc.reset_index(drop=True),ml,vl,genes

def all_effects(d,E,num,cat):
    genes=E.columns.tolist(); X=rv.design(d,'type',num,cat)
    y=rv.residualize_rank(d.rheobase.to_numpy(),X)[:,0]; y=(y-y.mean())/y.std(ddof=1)
    # Chunking avoids a large rank-residual matrix without changing estimates.
    out=[]
    for i in range(0,len(genes),1000):
        g=genes[i:i+1000]; R=rv.residualize_rank(E[g].to_numpy(),X); sd=R.std(0,ddof=1)
        R=(R-R.mean(0))/np.where(sd>0,sd,np.nan); out.extend((R*y[:,None]).sum(0)/(len(d)-1))
    return pd.Series(out,index=genes,dtype=float)

def main():
    m1,v1,mc,vc,ml,vl,genes=cohort_data()
    det=pd.DataFrame({'gene':genes,'detection_m1':(mc>0).mean(axis=0).values,'detection_v1':(vc>0).mean(axis=0).values,
                      'mean_logcpm_m1':ml.mean(axis=0).values,'mean_logcpm_v1':vl.mean(axis=0).values})
    det['eligible']=det.detection_m1.ge(.25)&det.detection_v1.ge(.25)
    panel_genes=set(sum(PANELS.values(),[])); det['panel_memberships']=det.gene.map(lambda g:';'.join(k for k,v in PANELS.items() if g in v))
    det.to_csv(TAB/'scaffolding_gene_definition_and_matching.csv',index=False)
    eligible=det.loc[det.eligible,'gene'].tolist(); me=all_effects(m1,ml[eligible],['log_libsize','n_detected','hold','age'],['batch','sex']); ve=all_effects(v1,vl[eligible],['log_libsize','n_detected','top50_share'],[])
    allper=pd.DataFrame({'gene':eligible,'effect_m1':me.reindex(eligible).values,'effect_v1':ve.reindex(eligible).values})
    allper.to_csv(TAB/'scaffolding_universe_per_gene_effects.csv',index=False)
    feat=det.set_index('gene').loc[eligible,['detection_m1','detection_v1','mean_logcpm_m1','mean_logcpm_v1']]
    Z=(feat-feat.mean())/feat.std().replace(0,np.nan); candidates=[g for g in eligible if g not in panel_genes]
    rows=[]; nullrows=[]; per=[]
    for pname,declared in PANELS.items():
        pg=[g for g in declared if g in eligible]
        for cohort,d,E,num,cat,effects in [('m1',m1,ml,['log_libsize','n_detected','hold','age'],['batch','sex'],me),('v1',v1,vl,['log_libsize','n_detected','top50_share'],[],ve)]:
            eff=float(effects.loc[pg].mean()); boot,ci,sp=rv.cluster_boot(d,E,pg,'type',num,cat,nboot=1000)
            # Candidate neighbourhoods are fixed by four outcome-free expression metrics.
            neighbourhood={g:((Z.loc[candidates]-Z.loc[g])**2).sum(axis=1).nsmallest(50).index.to_list() for g in pg}
            null=[]
            for rep in range(5000):
                chosen=[]
                for g in pg:
                    pool=[x for x in neighbourhood[g] if x not in chosen]; chosen.append(RNG.choice(pool if pool else neighbourhood[g]))
                null.append(float(effects.loc[chosen].mean()))
            null=np.asarray(null); specificity=float((np.abs(null)>=abs(eff)).mean())
            rows.append({'cohort':cohort,'panel':pname,'declared_genes':len(declared),'retained_genes':len(pg),'effect':eff,'ci_lo':ci[0],'ci_hi':ci[1],'descriptive_bootstrap_sign_probability':sp,'matched_random_set_two_sided_tail_fraction':specificity,'status':'post_validation_exploratory'})
            nullrows.extend({'cohort':cohort,'panel':pname,'replicate':i,'effect':x} for i,x in enumerate(null))
            per.extend({'cohort':cohort,'panel':pname,'gene':g,'effect':effects[g]} for g in pg)
    pd.DataFrame(rows).to_csv(TAB/'scaffolding_panel_results.csv',index=False)
    pd.DataFrame(nullrows).to_csv(TAB/'scaffolding_matched_random_set_null.csv',index=False)
    pd.DataFrame(per).to_csv(TAB/'scaffolding_panel_per_gene_effects.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
