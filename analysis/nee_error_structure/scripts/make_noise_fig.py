#!/usr/bin/env python
import os, math
import numpy as np, pandas as pd, torch
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE="/tmp/claude-1000/-home-cognitia-Desktop-Work-PhD-WienerNet/b4a3f217-adb5-459c-9476-66908216cac1/scratchpad/nllcmp"
OUT=os.path.dirname(os.path.abspath(__file__))
TREF,T0=10.0,46.02
C_ST,C_BE,C_EMP,C_GAU,C_LAP="#0072B2","#009E73","#111111","#7a7a7a","#D55E00"
def reco(Ta,E0,rb): return rb*np.exp(E0*(1/(TREF+T0)-1/(Ta+T0)))
def lb(x): return np.mean(np.abs(x-np.median(x)))
def nu_of(m):
    ck=torch.load(f"{BASE}/{m}/checkpoints/best.pth",map_location="cpu",weights_only=False)
    sd=ck.get("model_state_dict",ck)
    return float(math.log1p(math.exp(float(sd["log_nu"])))+1) if "log_nu" in sd else np.inf

D={}
for m in ["nll_student_t","nll_beta"]:
    pp=pd.read_parquet(f"{BASE}/{m}/metrics/predictions.parquet")
    flux=reco(pp.Ta.values,pp.E0.values,pp.rb.values)
    emp=(pp.NEE_next.values-pp.NEE.values)-pp.pred_f.values*pp.dt.values
    ps=pp.pred_noise_stds.values*np.sqrt(pp.dt.values)
    g=np.isfinite(flux)&np.isfinite(emp)&np.isfinite(ps)&(ps>0)
    D[m]=dict(flux=flux[g],emp=emp[g],ps=ps[g],nu=nu_of(m))

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.grid":True,
  "grid.color":"#e8e8e8","grid.linewidth":.7,"axes.axisbelow":True,
  "axes.edgecolor":"#888","axes.linewidth":.8})
fig,ax=plt.subplots(1,2,figsize=(11,4.5))

# Panel A: scale vs flux
a=ax[0]
f=D["nll_beta"]["flux"]
qs=np.unique(np.quantile(f,np.linspace(0,1,17)))
idx=np.clip(np.digitize(f,qs[1:-1]),0,len(qs)-2)
cen=np.array([np.median(f[idx==b]) for b in range(len(qs)-1)])
# empirical robust scale (shared data)
empb=np.array([lb(D["nll_beta"]["emp"][idx==b])/0.6745 for b in range(len(qs)-1)])  # ~sd via MAD
a.plot(cen,empb,"o-",color=C_EMP,lw=2,ms=5,zorder=5,label="Empirical residual scale")
for m,c,l in [("nll_student_t",C_ST,"Student-t  σ·√dt"),("nll_beta",C_BE,"β-NLL  σ·√dt")]:
    fm=D[m]["flux"]; im=np.clip(np.digitize(fm,qs[1:-1]),0,len(qs)-2)
    pm=np.array([np.mean(D[m]["ps"][im==b]) for b in range(len(qs)-1)])
    a.plot(cen,pm,"s--",color=c,lw=1.8,ms=4,label=l)
a.plot(cen,0.30*cen,":",color=C_LAP,lw=1.6,label="0.30·flux (report level-slope)")
a.set_xlabel("Predicted nighttime flux  Reco  (µmol m⁻² s⁻¹)")
a.set_ylabel("Noise scale (increment units)")
a.set_title("A  Predicted σ is heteroscedastic but too shallow",loc="left",fontweight="bold",fontsize=11)
a.legend(frameon=False,fontsize=8.5,loc="upper left")

# Panel B: whitened tail
b=ax[1]
ks=np.linspace(1,6,60)
b.plot(ks,2*stats.norm.sf(ks),color=C_GAU,lw=1.8,ls="--",label="Gaussian")
b.plot(ks,np.exp(-np.sqrt(2)*ks),color=C_LAP,lw=1.8,ls="--",label="Laplace")
for m,c,lab in [("nll_student_t",C_ST,f"Student-t whitened (ν={D['nll_student_t']['nu']:.1f})"),
                ("nll_beta",C_BE,"β-NLL whitened (Gaussian)")]:
    z=D[m]["emp"]/D[m]["ps"]; zc=z/np.std(z)
    b.plot(ks,[np.mean(np.abs(zc)>k) for k in ks],color=c,lw=2.2,label=lab)
nu=D["nll_student_t"]["nu"]
b.plot(ks,[2*stats.t.sf(k*np.sqrt(nu/(nu-2)),df=nu) for k in ks],color=C_ST,lw=1.2,ls=":",
       label=f"t(ν={nu:.1f}) ideal")
b.axvline(3,color="#ccc",lw=.8,ls=":")
b.set_yscale("log"); b.set_ylim(1e-4,1)
b.set_xlabel("Threshold k  (standard deviations)")
b.set_ylabel("P(|z| > k)  after whitening by model σ")
b.set_title("B  Whitened residual still heavier than the model assumes",loc="left",fontweight="bold",fontsize=11)
b.legend(frameon=False,fontsize=8.5)

fig.tight_layout(pad=1.3)
fig.savefig(os.path.join(OUT,"fig_noise_head.png"),dpi=170,bbox_inches="tight")
print("wrote fig_noise_head.png")
