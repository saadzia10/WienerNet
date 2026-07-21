#!/usr/bin/env python
"""Structural-error handling for the gap-fill band.

The physics level Reco(T_air) omits the soil-temperature/moisture drivers of respiration, so
the observed flux scatters around the reconstructed level by more than the measurement noise
the aleatoric heads model. Two components are provided:

  A. Reconstruction MEAN augmented with soil temperature (Tsoil1, the soil driver available
     at all five sites): NEE ~ a + b*Reco(Ta) + c*Tsoil1.

  B. An explicit structural-variance term sigma_struct added to the band in quadrature. It is
     FLAT in gap-time (a persistent missing-driver offset), unlike the process term which
     grows as sqrt(t). To avoid double-counting the measurement noise the aleatoric head
     already carries, sigma_struct^2 = max(0, Var(level residual) - sigma_meas^2).

Both are estimated on TRAIN (the sites the model saw) and applied to the held-out TEST site.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TREF, T0 = 10.0, 46.02


def reco(T, E0, rb):
    return rb * np.exp(E0 * (1.0 / (TREF + T0) - 1.0 / (T + T0)))


def _cols_ok(df, use_soil):
    need = ["NEE", "Ta", "E0", "rb"] + (["Tsoil1"] if use_soil else [])
    return all(c in df.columns for c in need)


def fit_level_model(train_df, use_soil=True):
    """Fit the reconstruction MEAN on the TRAIN sites.

    Returns a dict with the fitted coefficients, a `predict(df)` closure, the train
    level-residual std, and whether soil temperature was actually used (falls back to the
    Reco-only model if Tsoil1 is missing/degenerate at a site).
    """
    use_soil = bool(use_soil) and "Tsoil1" in train_df.columns
    d = train_df.dropna(subset=["NEE", "Ta", "E0", "rb"] + (["Tsoil1"] if use_soil else [])).copy()
    rc = reco(d["Ta"].values, d["E0"].values, d["rb"].values)
    if use_soil and np.nanstd(d["Tsoil1"].values) > 1e-6:
        X = np.column_stack([np.ones(len(d)), rc, d["Tsoil1"].values])
        beta, *_ = np.linalg.lstsq(X, d["NEE"].values, rcond=None)
        a, b, c = beta
        soil_used = True
    else:
        b, a = np.polyfit(rc, d["NEE"].values, 1)
        c = 0.0
        soil_used = False

    def predict(df):
        r = reco(df["Ta"].values, df["E0"].values, df["rb"].values)
        s = df["Tsoil1"].values if soil_used and "Tsoil1" in df.columns else 0.0
        return a + b * r + c * s

    resid = d["NEE"].values - predict(d)
    return dict(a=float(a), b=float(b), c=float(c), soil_used=soil_used,
                predict=predict, resid_std=float(np.nanstd(resid)))


def structural_sigma(train_df, sigma_meas_bar, use_soil=True):
    """sigma_struct estimated on TRAIN: the level-residual spread beyond measurement noise.

    sigma_struct^2 = max(0, Var(NEE - reconstructed_mean) - sigma_meas^2), so the band's
    flat term does not double-count the measurement noise the aleatoric head already carries.
    Returns (sigma_struct, level_model_dict).
    """
    lm = fit_level_model(train_df, use_soil=use_soil)
    var_resid = lm["resid_std"] ** 2
    sm2 = float(np.nan_to_num(sigma_meas_bar)) ** 2
    sigma_struct = float(np.sqrt(max(0.0, var_resid - sm2)))
    return sigma_struct, lm
