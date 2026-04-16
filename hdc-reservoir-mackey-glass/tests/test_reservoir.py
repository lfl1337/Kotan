"""Tests für das Reservoir-Modul (ESN und Dynamik-Funktionen)."""

import numpy as np
import pytest
from scipy import sparse

from src.reservoir.dynamics import (
    compute_spectral_radius,
    generate_sparse_matrix,
    scale_spectral_radius,
)
from src.reservoir.esn import EchoStateNetwork


# --- dynamics.py Tests ---

def test_generate_sparse_matrix_shape():
    W = generate_sparse_matrix(size=100, sparsity=0.1)
    assert W.shape == (100, 100)


def test_generate_sparse_matrix_is_sparse():
    W = generate_sparse_matrix(size=100, sparsity=0.1)
    assert sparse.issparse(W)


def test_generate_sparse_matrix_sparsity():
    W = generate_sparse_matrix(size=200, sparsity=0.1, seed=42)
    density = W.nnz / (200 * 200)
    assert abs(density - 0.1) < 0.02  # Toleranz ±2%


def test_generate_sparse_matrix_reproducible():
    W1 = generate_sparse_matrix(size=100, sparsity=0.1, seed=42)
    W2 = generate_sparse_matrix(size=100, sparsity=0.1, seed=42)
    assert (W1 - W2).nnz == 0


def test_compute_spectral_radius():
    W = generate_sparse_matrix(size=50, sparsity=0.2, seed=0)
    r = compute_spectral_radius(W)
    assert isinstance(r, float)
    assert r > 0


def test_scale_spectral_radius():
    W = generate_sparse_matrix(size=100, sparsity=0.1, seed=1)
    target = 0.9
    W_scaled = scale_spectral_radius(W, target)
    actual = compute_spectral_radius(W_scaled)
    assert abs(actual - target) < 1e-6


# --- EchoStateNetwork Tests ---

def test_esn_state_shape():
    esn = EchoStateNetwork(n_reservoir=200, seed=42)
    states = esn.run(np.random.rand(100))
    assert states.shape == (100, 200)


def test_esn_spectral_radius_close_to_target():
    esn = EchoStateNetwork(n_reservoir=100, spectral_radius=0.85, seed=0)
    actual = compute_spectral_radius(esn.W_reservoir)
    assert abs(actual - 0.85) < 1e-5


def test_esn_reset_zeroes_state():
    esn = EchoStateNetwork(n_reservoir=100, seed=1)
    esn.run(np.random.rand(50))
    esn.reset()
    assert np.all(esn.state == 0.0)


def test_esn_reproducible_with_seed():
    inputs = np.random.rand(100)
    esn1 = EchoStateNetwork(n_reservoir=100, seed=7)
    esn2 = EchoStateNetwork(n_reservoir=100, seed=7)
    states1 = esn1.run(inputs)
    states2 = esn2.run(inputs)
    np.testing.assert_array_equal(states1, states2)
