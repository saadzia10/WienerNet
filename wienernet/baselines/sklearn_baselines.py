"""sklearn / XGBoost baselines.

These don't fit the WienerNetModel API (no torch, no SDE physics) but share
the data pipeline + evaluation flow. Each class exposes:

    .fit(X_train, y_train)
    .predict(X_test) -> 1D numpy array
    .save(path), .load(path)
    .extract_X_y(bundle, split) -> (X_flat, y) for the flat feature matrix
        the baselines train on

So `scripts/train_baseline.py` can treat both uniformly.

Flat feature matrix matches the original training notebook's RF/XGB cells:
    columns = X (scaled drivers + season + time_vars + site_xyz)
              + bNEE      (current NEE as a feature)
              + E0, rb    (Lloyd-Taylor params)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - import guard
    XGBRegressor = None  # type: ignore[misc]

from ..data import DataBundle

log = logging.getLogger("wienernet.baselines")


# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------

class BaselineLike(Protocol):
    """Minimal interface every baseline must provide."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineLike": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


def extract_flat_features(
    bundle: DataBundle, split: Literal["train", "test"],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the flat feature matrix the baselines train on.

    Same recipe as the original training notebook cell 48 (RF) and cell 54 (XGB):
    concatenate the scaled features X with the scalar bNEE and the (E0, rb) pair,
    then predict the next-step NEE.
    """
    dataset = bundle.train_dataset if split == "train" else bundle.test_dataset
    df = bundle.train_df if split == "train" else bundle.test_df

    X = dataset.X.numpy()
    bNEE = dataset.bNEE.numpy().reshape(-1, 1)
    k = dataset.k.numpy()
    flat = np.concatenate([X, bNEE, k], axis=1)
    y = dataset.NEE.numpy()
    return flat, y


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

@dataclass
class RandomForestBaseline:
    """sklearn RandomForestRegressor for one-step NEE prediction.

    Matches the original notebook's `RandomForestRegressor(n_estimators=100, n_jobs=-1)`.
    All hyperparameters are configurable via Hydra.
    """

    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | float | None = 1.0
    n_jobs: int = -1
    random_state: int = 42
    bootstrap: bool = True

    _model: RandomForestRegressor | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        log.info("fitting RandomForestRegressor: n_estimators=%d, n_jobs=%d",
                 self.n_estimators, self.n_jobs)
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            bootstrap=self.bootstrap,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit() first")
        return self._model.feature_importances_

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestBaseline":
        inst = cls()
        inst._model = joblib.load(path)
        return inst


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

@dataclass
class XGBoostBaseline:
    """xgboost.XGBRegressor for one-step NEE prediction.

    Matches the original notebook's `XGBRegressor(n_estimators=100)`.
    """

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.3
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    n_jobs: int = -1
    random_state: int = 42
    tree_method: str = "hist"

    _model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if XGBRegressor is None:
            raise ImportError(
                "xgboost is not installed. Install with `pip install xgboost`."
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        log.info("fitting XGBRegressor: n_estimators=%d, lr=%g, max_depth=%d",
                 self.n_estimators, self.learning_rate, self.max_depth)
        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            tree_method=self.tree_method,
            verbosity=0,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit() first")
        return self._model.feature_importances_

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostBaseline":
        inst = cls.__new__(cls)
        inst.n_estimators = 100
        inst.max_depth = 6
        inst.learning_rate = 0.3
        inst.subsample = 1.0
        inst.colsample_bytree = 1.0
        inst.reg_alpha = 0.0
        inst.reg_lambda = 1.0
        inst.n_jobs = -1
        inst.random_state = 42
        inst.tree_method = "hist"
        inst._model = joblib.load(path)
        return inst


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_baseline(variant: str, **kwargs) -> BaselineLike:
    """Construct an unfitted baseline by name."""
    if variant == "rf":
        return RandomForestBaseline(**kwargs)
    if variant == "xgb":
        return XGBoostBaseline(**kwargs)
    raise ValueError(f"Unknown baseline variant {variant!r}; valid: ['rf', 'xgb']")
