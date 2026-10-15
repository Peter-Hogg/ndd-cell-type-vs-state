#!/usr/bin/env python3
"""Exploratory channel/receptor gene-family set effects on rheobase in the M1 and V1 cohorts.

Gene families follow the CHANNEL_PATTERNS annotation, split (independent of
outcomes) into ion-channel/auxiliary genes and neurotransmitter receptors; genes
detected in >= 25% of cells in both cohorts are kept. Uses the set statistic and
animal-cluster bootstrap from run_validation.py.

Usage:  uv run python analysis/cross_dataset/m1_v1_validation/scripts/run_channel_receptor_exploration.py
"""
from __future__ import annotations
import importlib.util, re
from pathlib import Path
import os
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[4]
CODE=Path(__file__).resolve().parents[1]
VAL=Path(os.environ.get('NDD_VALIDATION_OUT',CODE)).resolve()
TAB=VAL/'tables'
SEED=20260803

spec=importlib.util.spec_from_file_location('validation',CODE/'scripts/run_validation.py')
rv=importlib.util.module_from_spec(spec); spec.loader.exec_module(rv)
rv.RNG=np.random.default_rng(SEED)

CHANNEL_RE=(r'^(Kcn|Scn|Cacn|Hcn|Clcn|Trp[acmvp]|Ano|Asic|Nalcn|Ryr|Itpr|Cnga|Cngb|Kcnip|Dpp6|Dpp10|Fgf1[1-4]$|Sh3kbp1$)')
RECEPTOR_RE=(r'^(P2rx|Gabr|Grin|Gria|Grik|Grm|Chrn|Htr3|Glr[ab])')
FAMILIES={'ion_channel_or_auxiliary':re.compile(CHANNEL_RE),'neurotransmitter_receptor':re.compile(RECEPTOR_RE)}

def family(g):
    for name,pat in FAMILIES.items():
        if pat.match(str(g)): return name
    return None

def gene_universes():
    mpath=ROOT/'PatchSeqExpression/GSE163764_m1_patchseq_exon_counts.csv'
    mg=pd.read_csv(mpath,usecols=[0]).iloc[:,0].astype(str)
    first=next(iter(sorted((ROOT/'ReplicationData/MouseV1_Sorensen2026/quant').glob('*.parquet'))))
    vg=pd.read_parquet(first,columns=['gene']).gene.astype(str)
    rows=[]
    for g in sorted(set(mg)&set(vg)):
        f=family(g)
        if f: rows.append((g,f))
    return pd.DataFrame(rows,columns=['gene','family'])

def load_m1(d,genes):
    ids=pd.Index(d.cell_id); path=ROOT/'PatchSeqExpression/GSE163764_m1_patchseq_exon_counts.csv'; parts=[]; lib=pd.Series(0.,index=ids)
    for ch in pd.read_csv(path,index_col=0,usecols=lambda x:x=='Unnamed: 0' or x in set(ids),chunksize=2000):
        ch.index=ch.index.astype(str); lib=lib.add(ch.sum(axis=0),fill_value=0); hit=ch.index.intersection(genes)
        if len(hit): parts.append(ch.loc[hit])
    c=pd.concat(parts).groupby(level=0).sum().reindex(genes,fill_value=0).T.reindex(ids)
    return c.reset_index(drop=True),np.log1p(c.div(lib.loc[ids],axis=0)*1e6).reset_index(drop=True)

def load_v1(d,genes):
    frames={}; libs={}; qdir=ROOT/'ReplicationData/MouseV1_Sorensen2026/quant'
    wanted=set(d.cell_id.astype(str))
    for p in sorted(qdir.glob('*.parquet')):
        if p.stem not in wanted: continue
        z=pd.read_parquet(p,columns=['gene','NumReads']); s=z.set_index('gene').NumReads; frames[p.stem]=s.reindex(genes,fill_value=0); libs[p.stem]=s.sum()
    c=pd.DataFrame(frames).T.reindex(d.cell_id.astype(str)).fillna(0)
    lib=pd.Series(libs).reindex(c.index)
    return c.reset_index(drop=True),np.log1p(c.div(lib,axis=0)*1e6).reset_index(drop=True)

def main():
    TAB.mkdir(parents=True,exist_ok=True)
    defs=gene_universes(); genes=defs.gene.tolist()
    m1,_,_,_=rv.load_m1(); v1,_,_,_=rv.load_v1()
    mc,me=load_m1(m1,genes); vc,ve=load_v1(v1,genes)
    det=defs.copy(); det['detection_fraction_m1']=[float((mc[g]>0).mean()) for g in genes]; det['detection_fraction_v1']=[float((vc[g]>0).mean()) for g in genes]
    det['included']=det.detection_fraction_m1.ge(.25)&det.detection_fraction_v1.ge(.25)
    det.to_csv(TAB/'channel_receptor_gene_definition.csv',index=False)
    rows=[]; per=[]
    configs=[('m1',m1,me,['log_libsize','n_detected','hold','age'],['batch','sex']),('v1',v1,ve,['log_libsize','n_detected','top50_share'],[])]
    for cohort,d,E,num,cat in configs:
        for fam in FAMILIES:
            gg=det.loc[det.family.eq(fam)&det.included,'gene'].tolist()
            eff,gene_eff,_=rv.set_effect(d,E,gg,'type',num,cat)
            boot,ci,sp=rv.cluster_boot(d,E,gg,'type',num,cat,nboot=1000)
            rows.append({'cohort':cohort,'family':fam,'cells':len(d),'animals':d.animal.nunique(),'genes':len(gg),'effect':eff,'ci_lo':ci[0],'ci_hi':ci[1],'descriptive_bootstrap_sign_probability':sp,'status':'exploratory_fixed_family'})
            per.extend({'cohort':cohort,'family':fam,'gene':g,'effect':v,'status':'descriptive_only'} for g,v in gene_eff.items())
            pd.DataFrame({'replicate':range(len(boot)),'effect':boot}).to_csv(TAB/f'{cohort}_{fam}_bootstrap_results.csv',index=False)
    pd.DataFrame(rows).to_csv(TAB/'channel_receptor_set_results.csv',index=False)
    pd.DataFrame(per).to_csv(TAB/'channel_receptor_per_gene_effects.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
