"""Tests für den Ridge Regression Readout."""

import numpy as np
import pytest

from src.baselines.ridge_readout import RidgeReadout


def _make_data(n=200, n_features=100, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    w_true = rng.standard_normal(n_features)
    y = X @ w_true + rng.standard_normal(n) * 0.01
    return X, y


def test_fit_sets_weights():
    readout = RidgeReadout(alpha=1e-6)
    X, y = _make_data()
    readout.fit(X, y)
    assert hasattr(readout, "W_out")
    assert readout.W_out.shape == (X.shape[1],)


def test_predict_shape():
    readout = RidgeReadout(alpha=1e-6)
    X, y = _make_data()
    readout.fit(X, y)
    preds = readout.predict(X)
    assert preds.shape == (len(y),)


def test_predict_low_error_on_linear_data():
    readout = RidgeReadout(alpha=1e-6)
    X, y = _make_data(n=500, n_features=50, seed=1)
    readout.fit(X, y)
    preds = readout.predict(X)
    rmse = np.sqrt(np.mean((preds - y) ** 2))
    assert rmse < 0.1


def test_score_returns_nrmse():
    readout = RidgeReadout(alpha=1e-6)
    X, y = _make_data(n=500, n_features=50, seed=2)
    readout.fit(X, y)
    nrmse = readout.score(X, y)
    assert isinstance(nrmse, float)
    assert nrmse >= 0


def test_score_nrmse_small_for_linear_data():
    readout = RidgeReadout(alpha=1e-6)
    X, y = _make_data(n=500, n_features=50, seed=3)
    readout.fit(X, y)
    nrmse = readout.score(X, y)
    assert nrmse < 0.1


def test_predict_before_fit_raises():
    readout = RidgeReadout()
    X, _ = _make_data()
    with pytest.raises(AttributeError):
        readout.predict(X)
