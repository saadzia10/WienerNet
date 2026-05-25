"""Tests for the sklearn / XGB baselines."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from wienernet.baselines import (
    RandomForestBaseline,
    XGBoostBaseline,
    build_baseline,
)


@pytest.fixture
def linear_dataset():
    rng = np.random.default_rng(0)
    n, d = 300, 5
    X = rng.standard_normal((n, d))
    true_w = rng.standard_normal(d)
    y = X @ true_w + rng.normal(0, 0.1, n)
    X_train, y_train = X[:240], y[:240]
    X_test, y_test = X[240:], y[240:]
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def test_rf_fits_and_predicts_correct_shape(linear_dataset):
    X_train, y_train, X_test, _ = linear_dataset
    model = RandomForestBaseline(n_estimators=10).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],)


def test_rf_learns_signal(linear_dataset):
    """On a linear signal RF should achieve much better than predicting the mean."""
    X_train, y_train, X_test, y_test = linear_dataset
    model = RandomForestBaseline(n_estimators=20).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse_pred = np.mean((y_test - y_pred) ** 2)
    mse_mean = np.mean((y_test - y_train.mean()) ** 2)
    assert mse_pred < mse_mean / 2, f"RF didn't beat mean baseline: {mse_pred} vs {mse_mean}"


def test_rf_feature_importance_sums_to_one(linear_dataset):
    X_train, y_train, _, _ = linear_dataset
    model = RandomForestBaseline(n_estimators=10).fit(X_train, y_train)
    imp = model.feature_importances_
    assert imp.shape == (X_train.shape[1],)
    assert np.isclose(imp.sum(), 1.0, atol=1e-6)


def test_rf_save_and_load_roundtrip(linear_dataset):
    X_train, y_train, X_test, _ = linear_dataset
    model = RandomForestBaseline(n_estimators=10).fit(X_train, y_train)
    y_pred_before = model.predict(X_test)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "rf.joblib"
        model.save(path)
        loaded = RandomForestBaseline.load(path)
    y_pred_after = loaded.predict(X_test)
    assert np.allclose(y_pred_before, y_pred_after)


def test_rf_deterministic_with_seed(linear_dataset):
    X_train, y_train, X_test, _ = linear_dataset
    a = RandomForestBaseline(n_estimators=10, random_state=7).fit(X_train, y_train).predict(X_test)
    b = RandomForestBaseline(n_estimators=10, random_state=7).fit(X_train, y_train).predict(X_test)
    assert np.allclose(a, b)


def test_rf_raises_before_fit():
    model = RandomForestBaseline()
    with pytest.raises(RuntimeError, match="Call fit"):
        model.predict(np.zeros((1, 5)))


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def test_xgb_fits_and_predicts_correct_shape(linear_dataset):
    X_train, y_train, X_test, _ = linear_dataset
    model = XGBoostBaseline(n_estimators=10).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],)


def test_xgb_learns_signal(linear_dataset):
    X_train, y_train, X_test, y_test = linear_dataset
    model = XGBoostBaseline(n_estimators=50).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse_pred = np.mean((y_test - y_pred) ** 2)
    mse_mean = np.mean((y_test - y_train.mean()) ** 2)
    assert mse_pred < mse_mean / 2


def test_xgb_save_and_load_roundtrip(linear_dataset):
    X_train, y_train, X_test, _ = linear_dataset
    model = XGBoostBaseline(n_estimators=10).fit(X_train, y_train)
    y_pred_before = model.predict(X_test)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "xgb.joblib"
        model.save(path)
        loaded = XGBoostBaseline.load(path)
    y_pred_after = loaded.predict(X_test)
    assert np.allclose(y_pred_before, y_pred_after)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_build_baseline_factory_returns_correct_types():
    assert isinstance(build_baseline("rf"), RandomForestBaseline)
    assert isinstance(build_baseline("xgb"), XGBoostBaseline)


def test_build_baseline_factory_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown baseline variant"):
        build_baseline("svm")


def test_build_baseline_passes_kwargs():
    rf = build_baseline("rf", n_estimators=5, max_depth=3)
    assert rf.n_estimators == 5
    assert rf.max_depth == 3
