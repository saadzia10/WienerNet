"""Aggregated-scale (weekly/monthly) physics vs MDN on held-out great_fen (predicted-k)."""
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
R=Path("outputs/holdout_gf_predk"); OUT=Path("analysis/holdout_extrapolation/figures"); SEEDS=[42,0,1]
def a(m,s): 
    p=R/f"{m}_f100_s{s}/metrics/aggregated.json"; return json.load(open(p)) if p.exists() else None
models={"increment_B_predk":("Physics (predicts k)","#009E73","s"),"hetero_mdn":("No-physics MDN","#CC79A7","D")}
fig,ax=plt.subplots(1,2,figsize=(11,4.4))
# noise-cancellation curve
xs=[0.5,7*24*0.5/24*7,30]  # rough scale marker (raw~0.5h, week, month) -> use categorical
labels=["raw (0.5h)","weekly","monthly"]
for m,(lab,c,mk) in models.items():
    raw=np.mean([a(m,s)["raw_rel_uncertainty"] for s in SEEDS])
    wk=np.mean([a(m,s)["weekly"]["rel_uncertainty"] for s in SEEDS])
    mo=np.mean([a(m,s)["monthly"]["rel_uncertainty"] for s in SEEDS])
    ax[0].plot([0,1,2],[raw,wk,mo],mk+"-",color=c,label=lab,lw=1.8,ms=7)
ax[0].set_xticks([0,1,2]); ax[0].set_xticklabels(labels); ax[0].set_yscale("log")
ax[0].set_ylabel("relative uncertainty (log)"); ax[0].set_title("Noise cancels under aggregation (~1/√N)"); ax[0].grid(alpha=.3)
ax[0].legend(fontsize=8,frameon=False)
# aggregated R2 + cov90 across fractions (weekly)
fr=[("f100",49378),("f050",24689),("f025",12344),("f010",4938),("f005",2469),("f002",988)]
rows=[n for _,n in fr]
def wk(m,code,path):
    vs=[]
    for s in SEEDS:
        p=R/f"{m}_{code}_s{s}/metrics/aggregated.json"
        if not p.exists(): continue
        o=json.load(open(p)).get("weekly",{})
        for k in path: o=o.get(k) if isinstance(o,dict) else None
        vs.append(o)
    vs=np.array([v for v in vs if v is not None],float); return np.nanmean(vs),np.nanstd(vs)
for m,(lab,c,mk) in models.items():
    cv=[wk(m,code,["ensemble","coverage","0.90","coverage"]) for code,_ in fr]
    ax[1].errorbar(rows,[x[0] for x in cv],yerr=[x[1] for x in cv],fmt=mk+"-",color=c,label=lab,lw=1.8,ms=6,capsize=3)
ax[1].axhline(0.90,ls="--",c="gray",lw=1); ax[1].set_xscale("log")
ax[1].set_xlabel("training rows"); ax[1].set_ylabel("weekly 90% coverage")
ax[1].set_title("Aggregated calibration (physics over-disperses at high data)"); ax[1].grid(alpha=.3)
fig.suptitle("Weekly/monthly aggregated: physics vs MDN, held-out Great Fen (predicted-k, 3 seeds)",fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"holdout_aggregated.png",dpi=140,bbox_inches="tight")
print("wrote",OUT/"holdout_aggregated.png")
