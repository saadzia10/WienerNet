#!/usr/bin/env python
"""Test whether the physics residual (NEE_obs - NEE_phys) is heteroscedastic,
and whether it is Gaussian or double-exponential (Laplace).

Background (Hollinger & Richardson 2005; Richardson et al. 2006, 2008):
 * The random flux error is NOT Gaussian — it is double-exponential (Laplace),
   i.e. leptokurtic with heavy tails.
 * It is heteroscedastic: the error standard deviation scales ~linearly with the
   flux magnitude (different slopes for uptake vs release).
 * Random errors are near-uncorrelated in time, so they cancel ~1/sqrt(N) under
   aggregation: half-hourly relative uncertainty ~100% collapses to ~10% at the
   annual sum. Systematic (structural) error does NOT cancel.

Here NEE_phys = Lloyd-Taylor Reco(Ta; E0, rb) (nighttime, GPP=0), which is
exactly the model whose residuals Hollinger/Richardson used to characterise the
error. So residual = NEE_obs - Reco conflates random measurement error with
model structural error; we separate the systematic (mean) part from the random
(scatter) part where it matters.

Run from repo root:
    conda activate pytorch
    python residual_error_structure.py
"""
from __future__ import annotations

import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

SITE_FILES = {
    "rosedene": "data_manipulation/final_night_data.parquet",
    **{
        os.path.basename(os.path.dirname(p)): p
        for p in sorted(glob.glob("data_manipulation/other_sites/*/final_night_data.parquet"))
    },
}
TREF, T0 = 10.0, 46.02
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 500)


def reco(Ta, E0, rb):
    return rb * np.exp(E0 * (1.0 / (TREF + T0) - 1.0 / (Ta + T0)))


def load_residuals() -> dict[str, pd.DataFrame]:
    out = {}
    for site, path in SITE_FILES.items():
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        need = {"NEE", "Ta", "E0", "rb"}
        if not need <= set(df.columns):
            continue
        d = df[["DateTime", "NEE", "Ta", "E0", "rb"]].copy()
        d["phy"] = reco(d.Ta.values, d.E0.values, d.rb.values)
        d["resid"] = d.NEE - d.phy
        d = d[np.isfinite(d.resid) & np.isfinite(d.phy)]
        out[site] = d.sort_values("DateTime").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Shape: Gaussian vs Laplace on a (centered) sample
# ---------------------------------------------------------------------------
def shape_report(x: np.ndarray, label: str) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    # Gaussian MLE
    mu, sig = x.mean(), x.std(ddof=0)
    ll_norm = np.sum(stats.norm.logpdf(x, mu, sig))
    # Laplace MLE (loc=median, scale=mean|x-median|)
    loc = np.median(x)
    b = np.mean(np.abs(x - loc))
    ll_lap = np.sum(stats.laplace.logpdf(x, loc, b))
    aic_norm = 2 * 2 - 2 * ll_norm
    aic_lap = 2 * 2 - 2 * ll_lap
    exkurt = stats.kurtosis(x, fisher=True)   # 0=Gaussian, 3=Laplace
    skew = stats.skew(x)
    # empirical tail exceedance vs the two fitted models (in std units)
    z = (x - mu) / sig
    tails = {}
    for k in (2, 3, 4, 5):
        emp = np.mean(np.abs(z) > k)
        pg = 2 * stats.norm.sf(k)
        # Laplace with same variance: scale = sig/sqrt(2); P(|x-loc|>k*sig)
        pl = np.exp(-k * sig / (sig / np.sqrt(2)))  # = exp(-sqrt(2)*k)
        tails[k] = (emp, pg, pl)
    return dict(label=label, n=n, skew=skew, exkurt=exkurt,
                dAIC_normminuslap=aic_norm - aic_lap,
                ll_norm=ll_norm, ll_lap=ll_lap, tails=tails, sig=sig, b=b)


def print_shape(rep: dict):
    print(f"\n[{rep['label']}]  n={rep['n']}")
    print(f"   skew={rep['skew']:+.3f}   excess_kurtosis={rep['exkurt']:+.2f}  "
          f"(Gaussian=0, Laplace=+3)")
    winner = "LAPLACE" if rep["dAIC_normminuslap"] > 0 else "GAUSSIAN"
    print(f"   AIC(Normal)-AIC(Laplace) = {rep['dAIC_normminuslap']:+.1f}  "
          f"-> better fit: {winner}  (positive => Laplace)")
    print(f"   tail exceedance P(|z|>k):   "
          f"{'k':>2} {'empirical':>10} {'Gaussian':>10} {'Laplace':>10}")
    for k, (emp, pg, pl) in rep["tails"].items():
        print(f"                               {k:>2} {emp:10.4f} {pg:10.4f} {pl:10.4f}")


# ---------------------------------------------------------------------------
# Heteroscedasticity: does residual scale grow with flux magnitude?
# ---------------------------------------------------------------------------
def hetero_report(d: pd.DataFrame, nbins=16):
    x = d.phy.values
    r = d.resid.values
    m = np.isfinite(x) & np.isfinite(r)
    x, r = x[m], r[m]
    # bin by predicted flux quantiles
    qs = np.quantile(x, np.linspace(0, 1, nbins + 1))
    qs = np.unique(qs)
    idx = np.clip(np.digitize(x, qs[1:-1]), 0, len(qs) - 2)
    rows = []
    for b in range(len(qs) - 1):
        sel = idx == b
        if sel.sum() < 30:
            continue
        rr = r[sel]
        rows.append(dict(
            flux_center=np.median(x[sel]),
            n=int(sel.sum()),
            bias=np.median(rr),                       # systematic part
            sd=np.std(rr, ddof=1),                    # Gaussian scale
            laplace_b=np.mean(np.abs(rr - np.median(rr))),  # Laplace scale
        ))
    bt = pd.DataFrame(rows)
    # regress local scale on flux magnitude
    slope_sd = r2_sd = slope_b = np.nan
    if len(bt) >= 3:
        A = np.vstack([bt.flux_center, np.ones(len(bt))]).T
        (slope_sd, _), *_ = np.linalg.lstsq(A, bt.sd, rcond=None)
        pred = A @ np.linalg.lstsq(A, bt.sd, rcond=None)[0]
        r2_sd = 1 - np.sum((bt.sd - pred) ** 2) / np.sum((bt.sd - bt.sd.mean()) ** 2)
        (slope_b, _), *_ = np.linalg.lstsq(A, bt.laplace_b, rcond=None)
    # Breusch-Pagan-ish: corr(|resid|, flux)
    bp_r, bp_p = stats.pearsonr(np.abs(r), x)
    return bt, dict(slope_sd=slope_sd, r2_sd=r2_sd, slope_b=slope_b,
                    absresid_flux_r=bp_r, absresid_flux_p=bp_p)


# ---------------------------------------------------------------------------
# Aggregation: random-error cancellation (~1/sqrt(N))
# ---------------------------------------------------------------------------
def cancellation_report(d: pd.DataFrame):
    """Contiguous block sums of size N: uncertainty of an N-step sum and its
    relative size vs the summed flux. iid random errors => sd(sum) ~ sd0*sqrt(N),
    so relative uncertainty ~ 1/sqrt(N)."""
    r = d.resid.values
    f = d.NEE.values  # observed flux we are summing
    n = len(r)
    lag1 = pd.Series(r).autocorr(lag=1)
    sd0 = np.std(r, ddof=1)
    rows = []
    for N, lbl in [(1, "30-min"), (2, "1 h"), (4, "2 h"), (10, "5 h (~1 night)"),
                    (48, "~1 day-eq"), (336, "~1 week-eq"), (1440, "~1 month-eq"),
                    (17520, "~1 year-eq")]:
        if N > n:
            continue
        nb = n // N
        rb_ = r[: nb * N].reshape(nb, N).sum(axis=1)   # residual sum per block
        fb = f[: nb * N].reshape(nb, N).sum(axis=1)    # flux sum per block
        sd_sum = np.std(rb_, ddof=1)
        mean_absflux = np.mean(np.abs(fb))
        rows.append(dict(
            block=lbl, N=N, n_blocks=nb,
            sd_of_Nsum=sd_sum,
            sd_over_sqrtN=sd_sum / np.sqrt(N),          # flat if iid
            rel_uncert=sd_sum / mean_absflux if mean_absflux > 0 else np.nan,
        ))
    return pd.DataFrame(rows), dict(lag1_autocorr=lag1, sd0=sd0, n=n)


def fmt(df):
    return df.to_string(index=False,
                        float_format=lambda v: f"{v:9.4f}" if pd.notna(v) else "      NaN")


def main():
    data = load_residuals()
    pooled = pd.concat(data.values(), ignore_index=True) if data else pd.DataFrame()
    print("Sites with E0/rb (physics residual computable):",
          {s: len(d) for s, d in data.items()})
    print(f"POOLED n = {len(pooled)}")

    # =============================== SHAPE ================================
    print("\n" + "=" * 92)
    print("A. DISTRIBUTION SHAPE  — Gaussian vs double-exponential (Laplace)")
    print("=" * 92)
    r = pooled.resid.values
    r = r - np.median(r)   # remove systematic offset

    print("\n--- (A1) RAW residuals (heteroscedastic mixture of scales) ---")
    print_shape(shape_report(r, "raw residual, pooled"))

    # winsorize lightly to show it's not just a few spikes
    lo, hi = np.percentile(r, [0.5, 99.5])
    print("\n--- (A2) after 0.5/99.5% winsorising (spikes removed) ---")
    print_shape(shape_report(np.clip(r, lo, hi), "winsorised residual"))

    # locally standardised: divide each residual by its flux-bin scale, so the
    # heteroscedasticity is removed and we test the SHAPE alone.
    bt, _ = hetero_report(pooled)
    x = pooled.phy.values
    qs = np.quantile(x, np.linspace(0, 1, 17))
    qs = np.unique(qs)
    idx = np.clip(np.digitize(x, qs[1:-1]), 0, len(qs) - 2)
    binscale = pd.Series(pooled.resid.values).groupby(idx).transform(
        lambda s: np.mean(np.abs(s - np.median(s))))  # local Laplace scale b
    std_resid = (pooled.resid.values - np.median(pooled.resid.values)) / binscale.values
    std_resid = std_resid[np.isfinite(std_resid)]
    print("\n--- (A3) LOCALLY STANDARDISED residuals (heteroscedasticity removed) ---")
    print("    -> if STILL leptokurtic/Laplace, the double-exponential shape is intrinsic,")
    print("       not an artefact of mixing different flux magnitudes.")
    print_shape(shape_report(std_resid, "flux-bin standardised residual"))

    # formal normality tests on a subsample
    sub = np.random.default_rng(0).choice(std_resid, size=min(4000, std_resid.size),
                                           replace=False)
    W, pW = stats.shapiro(sub)
    ad = stats.anderson(std_resid, dist="norm")
    jb, pJB = stats.jarque_bera(std_resid)
    print(f"\n   Normality tests on standardised residuals:")
    print(f"     Shapiro-Wilk (n={sub.size}):  W={W:.4f}  p={pW:.2e}  "
          f"({'REJECT normal' if pW < 0.05 else 'cannot reject'})")
    print(f"     Anderson-Darling:  A2={ad.statistic:.2f}  crit@5%={ad.critical_values[2]:.3f}  "
          f"({'REJECT normal' if ad.statistic > ad.critical_values[2] else 'cannot reject'})")
    print(f"     Jarque-Bera:  JB={jb:.1f}  p={pJB:.2e}  "
          f"({'REJECT normal' if pJB < 0.05 else 'cannot reject'})")

    # ========================= HETEROSCEDASTICITY ========================
    print("\n" + "=" * 92)
    print("B. HETEROSCEDASTICITY  — does residual scale grow with flux magnitude?")
    print("=" * 92)
    bt, hs = hetero_report(pooled)
    print("\nResidual scale per predicted-flux bin (pooled):")
    print(fmt(bt))
    print(f"\n   sd(resid) ~ a + b*flux :  slope={hs['slope_sd']:.4f}  R^2={hs['r2_sd']:.3f}")
    print(f"   Laplace-scale slope    :  {hs['slope_b']:.4f}")
    print(f"   corr(|resid|, flux)    :  r={hs['absresid_flux_r']:.3f}  "
          f"p={hs['absresid_flux_p']:.1e}   (Breusch-Pagan-style; >0 => heteroscedastic)")
    print("\n   Per-site slope of sd(resid) vs flux:")
    for s, d in data.items():
        _, hi_ = hetero_report(d)
        print(f"     {s:12s} slope_sd={hi_['slope_sd']:.4f}  R2={hi_['r2_sd']:.3f}  "
              f"corr(|r|,flux)={hi_['absresid_flux_r']:.3f}")

    # ========================== CANCELLATION =============================
    print("\n" + "=" * 92)
    print("C. TEMPORAL AGGREGATION  — do random errors cancel ~1/sqrt(N)?")
    print("=" * 92)
    ct, ci = cancellation_report(pooled.sort_values('DateTime'))
    print(f"\n   lag-1 autocorrelation of half-hourly residual = {ci['lag1_autocorr']:.3f} "
          f"(near 0 => errors ~independent => sqrt(N) cancellation valid)")
    print(f"   base half-hourly residual sd (sd0) = {ci['sd0']:.3f}\n")
    print("   Block-sum uncertainty vs block size N (contiguous):")
    print(fmt(ct))
    print("\n   Read: sd_over_sqrtN ~ flat confirms iid-like cancellation;")
    print("         rel_uncert = sd(N-sum)/mean|flux N-sum| should fall from ~1 (100%) toward ~0.1 (10%).")

    print("\n" + "=" * 92)
    print("CAVEAT: residual = measurement error + model structural error. The")
    print("systematic (bias) part does NOT cancel under aggregation; only the")
    print("random part does. Bias per bin is the 'bias' column in section B.")
    print("=" * 92)


if __name__ == "__main__":
    main()
