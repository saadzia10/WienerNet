"""Leave-one-site-out: GT-k physics vs MDN per held-out site (3 seeds)."""
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
R=Path("outputs/loso"); SEEDS=[42,0,1]
sites=["redmere_2","rosedene","great_fen","woodwalton","redmere_1"]  # easy->hard
labels=["Redmere 2","Rosedene","Great Fen","Woodwalton","Redmere 1"]
def crps(m,site):
    v=[]
    for s in SEEDS:
        p=R/f"{m}_{site}_s{s}/metrics/probabilistic.json"
        if p.exists(): v.append(json.load(open(p))["global"]["ensemble"]["crps"])
    v=np.array(v,float); return np.nanmean(v),np.nanstd(v)
fig,ax=plt.subplots(figsize=(8.5,4.6))
x=np.arange(len(sites)); w=0.36
for i,(m,lab,c) in enumerate([("increment_B_gtk","Physics (GT-k)","#009E73"),("hetero_mdn","No-physics MDN","#CC79A7")]):
    mu=[crps(m,s)[0] for s in sites]; sd=[crps(m,s)[1] for s in sites]
    ax.bar(x+(i-0.5)*w,mu,w,yerr=sd,capsize=3,color=c,label=lab)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("one-step ensemble CRPS (log) — lower better")
ax.set_title("Leave-one-site-out: physics vs MDN per held-out site (3 seeds)\nphysics wins Woodwalton, ties/loses middle, catastrophic on Redmere 1 (dTa-head OOD × drift amplification)",fontsize=9.5)
ax.legend(frameon=False); ax.grid(axis="y",alpha=.3)
fig.tight_layout(); fig.savefig("analysis/holdout_extrapolation/figures/loso_per_site.png",dpi=140,bbox_inches="tight")
print("wrote loso_per_site.png")
