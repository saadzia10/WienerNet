#!/usr/bin/env python
"""Test whether a trained noise head reproduces (i) the ~0.30 flux-slope
(heteroscedasticity) and (ii) the Laplace tail, for the student_t and beta
NLL runs.

The model is an INCREMENT SDE: NEE_{t+1} = NEE_t + f*dt + noise*sqrt(dt).
 * predicted per-min sigma  = pred_noise_stds        (the noise-head scale)
 * predicted increment scale s = sigma * sqrt(dt)     (t-scale / Gaussian-sigma)
 * increment residual to cover = (NEE_next - NEE) - f*dt
 * flux magnitude           = Reco(Ta;E0,rb)          (respiration level, as in the report)
Student-t dof: nu = softplus(log_nu)+1.
"""
from __future__ import annotations
import os, math
import numpy as np, pandas as pd, torch
from scipy import stats

BASE = "/tmp/claude-1000/-home-cognitia-Desktop-Work-PhD-WienerNet/b4a3f217-adb5-459c-9476-66908216cac1/scratchpad/nllcmp"
TREF, T0 = 10.0, 46.02
pd.set_option("display.width", 200)


def reco(Ta, E0, rb):
    return rb * np.exp(E0 * (1.0 / (TREF + T0) - 1.0 / (Ta + T0)))


def model_nu(method):
    ck = torch.load(f"{BASE}/{method}/checkpoints/best.pth", map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    if "log_nu" in sd:
        ln = float(sd["log_nu"])
        return float(math.log1p(math.exp(ln)) + 1.0)  # softplus+1
    return np.inf


def laplace_b(x):
    return np.mean(np.abs(x - np.median(x)))


def linfit(xc, yc):
    A = np.vstack([xc, np.ones(len(xc))]).T
    coef, *_ = np.linalg.lstsq(A, yc, rcond=None)
    pred = A @ coef
    r2 = 1 - np.sum((yc - pred) ** 2) / np.sum((yc - yc.mean()) ** 2)
    return coef[0], coef[1], r2


def analyse(method):
    nu = model_nu(method)
    pp = pd.read_parquet(f"{BASE}/{method}/metrics/predictions.parquet")
    Ta = pp.Ta.values; E0 = pp.E0.values; rb = pp.rb.values; dt = pp.dt.values
    flux = reco(Ta, E0, rb)
    dnee = pp.NEE_next.values - pp.NEE.values
    det = pp.pred_f.values * dt
    emp = dnee - det                                   # increment residual to cover
    sig = pp.pred_noise_stds.values                    # per-min sigma
    pred_scale = sig * np.sqrt(dt)                      # predicted increment scale
    good = np.isfinite(flux) & np.isfinite(emp) & np.isfinite(pred_scale) & (pred_scale > 0)
    flux, emp, pred_scale, sig = flux[good], emp[good], pred_scale[good], sig[good]
    n = len(flux)

    print("\n" + "#" * 90)
    print(f"# {method.upper()}   n={n}   learned nu = {nu:.2f}  "
          f"(Gaussian=inf, Laplace-equiv~ nu 2-4; excess kurt of t(nu)=6/(nu-4)={6/(nu-4):.2f} for nu>4)")
    print("#" * 90)

    # ---------- PART 1: heteroscedasticity of predicted scale ----------
    qs = np.unique(np.quantile(flux, np.linspace(0, 1, 17)))
    idx = np.clip(np.digitize(flux, qs[1:-1]), 0, len(qs) - 2)
    rows = []
    for b in range(len(qs) - 1):
        sel = idx == b
        if sel.sum() < 30:
            continue
        rows.append(dict(flux=np.median(flux[sel]),
                         emp_sd=np.std(emp[sel], ddof=1),
                         emp_b=laplace_b(emp[sel]),
                         pred_scale=np.mean(pred_scale[sel]),
                         sigma=np.mean(sig[sel]),
                         n=int(sel.sum())))
    bt = pd.DataFrame(rows)
    s_emp, _, r2_emp = linfit(bt.flux.values, bt.emp_sd.values)
    s_prd, _, r2_prd = linfit(bt.flux.values, bt.pred_scale.values)
    c_sig = np.corrcoef(sig, flux)[0, 1]
    c_prd = np.corrcoef(pred_scale, flux)[0, 1]

    print("\n[1] HETEROSCEDASTICITY — scale vs predicted flux (Reco), by flux bin")
    print(bt.assign(ratio=bt.pred_scale / bt.emp_sd).to_string(
        index=False, float_format=lambda v: f"{v:8.3f}"))
    print(f"\n    empirical increment-resid SD slope   = {s_emp:6.3f}  (R^2={r2_emp:.2f})")
    print(f"    MODEL predicted-scale slope          = {s_prd:6.3f}  (R^2={r2_prd:.2f})")
    print(f"    corr(predicted scale, flux)          = {c_prd:6.3f}")
    print(f"    corr(sigma, flux)                    = {c_sig:6.3f}")
    print(f"    reference: LEVEL-residual slope (report) = 0.30 ;  "
          f"increment ~ sqrt(2)x level ~ 0.42")
    het = ("HETEROSCEDASTIC (tracks flux)" if (s_prd > 0.05 and c_prd > 0.2)
           else "≈ HOMOSCEDASTIC (flux-independent)")
    print(f"    -> noise-head scale is: {het}")
    print(f"    -> scale calibration: overall pred/emp = "
          f"{pred_scale.mean()/np.std(emp,ddof=1):.2f} "
          f"(<1 => under-dispersed point scale; heavy tail can still fill variance)")

    # ---------- PART 2: tail shape of the model's noise ----------
    # (a) the model's own pull: z = emp / pred_scale ; should be standard t(nu) [student_t] or N(0,1) [beta]
    z = emp / pred_scale
    zc = z / np.std(z)                                  # unit-variance -> pure SHAPE test
    exk = stats.kurtosis(zc, fisher=True)
    tails = {}
    for k in (2, 3, 4, 5):
        emp_p = np.mean(np.abs(zc) > k)
        g = 2 * stats.norm.sf(k)
        lap = np.exp(-np.sqrt(2) * k)
        # standardised student-t(nu) with unit variance: scale by sqrt((nu-2)/nu)
        tnu = 2 * stats.t.sf(k * np.sqrt(nu / (nu - 2)), df=nu) if nu > 2 else np.nan
        tails[k] = (emp_p, g, lap, tnu)
    # implied nu the residual actually wants (fit t to locally-standardised resid)
    binb = pd.Series(emp).groupby(idx).transform(laplace_b).values
    zloc = (emp - np.median(emp)) / binb
    zloc = zloc[np.isfinite(zloc)]
    nu_hat, _, _ = stats.t.fit(zloc / np.std(zloc) * 1.0, floc=0)  # crude implied dof

    print("\n[2] TAIL SHAPE — whiten by model's predicted scale, then unit-variance shape test")
    print(f"    excess kurtosis of model-whitened residual = {exk:6.2f}  "
          f"(Gaussian 0, Laplace 3, t(nu={nu:.1f}) {6/(nu-4):.2f})")
    print(f"    {'k':>2} {'empirical':>10} {'Gaussian':>10} {'Laplace':>10} {'t(nu_model)':>12}")
    for k, (e, g, l, tnu) in tails.items():
        tn = f"{tnu:10.4f}" if np.isfinite(tnu) else "       n/a"
        print(f"    {k:>2} {e:10.4f} {g:10.4f} {l:10.4f} {tn:>12}")
    print(f"    implied dof the DATA tail wants (t.fit on locally-standardised resid) ~ nu={nu_hat:.2f}")
    verdict = ("captures heavy tail" if exk < 1.0 and tails[3][0] < 0.006
               else "LEAVES a heavy (Laplace-like) tail uncaptured")
    print(f"    -> {verdict}")
    return dict(method=method, nu=nu, n=n, s_emp=s_emp, s_prd=s_prd, c_prd=c_prd,
                exk=exk, tail3=tails[3][0], nu_hat=nu_hat,
                bt=bt, zc=zc, flux=flux, pred_scale=pred_scale, emp=emp)


def main():
    res = {}
    for m in ["nll_student_t", "nll_beta"]:
        res[m] = analyse(m)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    hdr = f"{'method':16s} {'nu':>7} {'pred-slope':>11} {'emp-slope':>10} {'corr(σ,flux)':>13} {'whiten exkurt':>14} {'P(|z|>3)':>9}"
    print(hdr)
    for m, r in res.items():
        print(f"{m:16s} {r['nu']:7.2f} {r['s_prd']:11.3f} {r['s_emp']:10.3f} "
              f"{r['c_prd']:13.3f} {r['exk']:14.2f} {r['tail3']:9.4f}")
    print("\nTargets from the empirical study:  flux-slope ≈ 0.30 (level) / ~0.42 (increment);")
    print("Laplace tail P(|z|>3) ≈ 0.014 with excess kurtosis ≈ 3 (or higher).")
    return res


if __name__ == "__main__":
    RES = main()
