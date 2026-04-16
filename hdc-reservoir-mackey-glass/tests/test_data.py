"""Tests für das Daten-Modul (Mackey-Glass-Generator)."""

import numpy as np
import pytest

from src.data.mackey_glass import generate_mackey_glass, prepare_dataset


def test_generate_default_length():
    data = generate_mackey_glass()
    assert len(data) == 10000


def test_generate_custom_length():
    data = generate_mackey_glass(n_steps=500)
    assert len(data) == 500


def test_generate_finite_values():
    data = generate_mackey_glass()
    assert np.all(np.isfinite(data))


def test_prepare_dataset_returns_four_arrays():
    data = generate_mackey_glass(n_steps=1000)
    result = prepare_dataset(data)
    assert len(result) == 4


def test_prepare_dataset_correct_lengths():
    data = generate_mackey_glass(n_steps=1000)
    X_train, y_train, X_test, y_test = prepare_dataset(data, train_ratio=0.8)
    total = len(X_train) + len(X_test)
    assert total == len(data) - 1  # prediction_horizon=1 kürzt um 1
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_prepare_dataset_no_overlap():
    data = generate_mackey_glass(n_steps=1000)
    X_train, _, X_test, _ = prepare_dataset(data, train_ratio=0.8)
    n_samples = len(data) - 1  # prediction_horizon=1
    expected_train = int(n_samples * 0.8)
    expected_test = n_samples - expected_train
    assert len(X_train) == expected_train
    assert len(X_test) == expected_test
