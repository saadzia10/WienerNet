"""Predicted-k held-out (great_fen): physics (estimates E0/rb) vs MDN, 3 seeds."""
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
R=Path("outputs/holdout_gf_predk"); OUT=Path("analysis/holdout_extrapolation/figures")
fr=[("f100",49378),("f050",24689),("f025",12344),("f010",4938),("f005",2469),("f002",988)]
rows=[n for _,n in fr]; SEEDS=[42,0,1]
def get(m,code,which):
    vs=[]
    for s in SEEDS:
        p=R/f"{m}_{code}_s{s}/metrics/probabilistic.json"
        if not p.exists(): continue
        j=json.load(open(p)); en=j["global"].get("ensemble") or {}
        if which=="crps": vs.append(en.get("crps"))
        else: vs.append((en.get("coverage") or {}).get("0.90",{}).get("coverage"))
    vs=np.array(vs,float); return np.nanmean(vs),np.nanstd(vs)
models={"increment_B_predk":("Increment-B (physics, predicts k)","#009E73","s"),
        "hetero_mdn":("No-physics MDN","#CC79A7","D")}
fig,ax=plt.subplots(1,2,figsize=(11,4.4))
for m,(lab,c,mk) in models.items():
    cr=[get(m,code,"crps") for code,_ in fr]; cv=[get(m,code,"cov") for code,_ in fr]
    ax[0].errorbar(rows,[x[0] for x in cr],yerr=[x[1] for x in cr],fmt=mk+"-",color=c,label=lab,lw=1.8,ms=6,capsize=3)
    ax[1].errorbar(rows,[x[0] for x in cv],yerr=[x[1] for x in cv],fmt=mk+"-",color=c,label=lab,lw=1.8,ms=6,capsize=3)
for a in ax: a.set_xscale("log"); a.set_xlabel("training rows (held-out site: Great Fen)"); a.grid(alpha=.3)
ax[0].set_ylabel("CRPS (ensemble) — lower better"); ax[0].set_title("Predictive skill (tied within noise)")
ax[1].axhline(0.90,ls="--",c="gray",lw=1); ax[1].set_ylabel("90% coverage"); ax[1].set_title("Calibration (physics modest edge)")
ax[0].legend(fontsize=8,frameon=False)
fig.suptitle("Predicted-k held-out extrapolation: physics estimating E0/rb vs MDN (k=1, 3 seeds ±std)",fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"holdout_predk_curve.png",dpi=140,bbox_inches="tight")
print("wrote",OUT/"holdout_predk_curve.png")
