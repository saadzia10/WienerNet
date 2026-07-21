"""Held-out-site (great_fen) data-efficiency curves: physics vs no-physics.
k=1, stabilized Student-t.
"""
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path("outputs/holdout_gf"); OUT=Path("analysis/holdout_extrapolation/figures")
fr=[("f100",49378),("f050",24689),("f025",12344),("f010",4938),("f005",2469),("f002",988)]
rows=[n for _,n in fr]
models={"increment_A":("Increment-A (physics+residual)","#0072B2","o"),
        "increment_B":("Increment-B (physics)","#009E73","s"),
        "hetero_mlp":("No-physics mean-var","#D55E00","^"),
        "hetero_mdn":("No-physics MDN","#CC79A7","D")}
def metric(m,code,which):
    j=json.load(open(ROOT/f"{m}_{code}_seed42/metrics/probabilistic.json"))
    g=j["global"]; en=g.get("ensemble") or {}
    if which=="crps": return en.get("crps") or g["probabilistic"]["crps"]
    if which=="cov90":
        c=en.get("coverage") or {}; return (c.get("0.90") or {}).get("coverage")
fig,ax=plt.subplots(1,2,figsize=(11,4.4))
for m,(lab,c,mk) in models.items():
    ax[0].plot(rows,[metric(m,code,"crps") for code,_ in fr],mk+"-",color=c,label=lab,lw=1.8,ms=6)
    ax[1].plot(rows,[metric(m,code,"cov90") for code,_ in fr],mk+"-",color=c,label=lab,lw=1.8,ms=6)
for a in ax: a.set_xscale("log"); a.set_xlabel("training rows (held-out site: Great Fen)"); a.grid(alpha=.3)
ax[0].set_ylabel("CRPS (ensemble) — lower better"); ax[0].set_title("Predictive skill vs training data")
ax[1].axhline(0.90,ls="--",c="gray",lw=1); ax[1].set_ylabel("90% coverage (ensemble)"); ax[1].set_title("Calibration vs training data (nominal 0.90)")
ax[0].legend(fontsize=8,frameon=False)
fig.suptitle("Held-out-site extrapolation: physics vs black box across training-data size (k=1, 1 seed)",fontsize=11)
fig.tight_layout(); fig.savefig(OUT/"holdout_learning_curve.png",dpi=140,bbox_inches="tight")
print("wrote",OUT/"holdout_learning_curve.png")
