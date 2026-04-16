"""Hilfsfunktionen für Reservoir-Dynamik.

Dieses Modul enthält Funktionen zur Erzeugung und Analyse
von Reservoir-Gewichtsmatrizen (Spektralradius-Skalierung,
Sparse-Matrix-Generierung, Stabilitätsanalyse).
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs


def generate_sparse_matrix(
    size: int,
    sparsity: float = 0.1,
    seed: int | None = None,
) -> sparse.csr_matrix:
    """Erzeugt eine sparse, zufällige Matrix."""
    rng = np.random.default_rng(seed)
    mask = rng.random((size, size)) < sparsity
    values = rng.standard_normal((size, size))
    dense = np.where(mask, values, 0.0)
    return sparse.csr_matrix(dense)


def scale_spectral_radius(
    matrix: sparse.csr_matrix,
    target_radius: float,
) -> sparse.csr_matrix:
    """Skaliert eine Matrix auf den gewünschten Spektralradius."""
    current = compute_spectral_radius(matrix)
    return matrix * (target_radius / current)


def compute_spectral_radius(matrix: sparse.csr_matrix) -> float:
    """Berechnet den Spektralradius (größter Eigenwert-Betrag) einer Matrix."""
    # k=1: nur der größte Eigenwert — schneller als full eigendecomposition
    eigenvalues = eigs(matrix.astype(float), k=1, which="LM", return_eigenvectors=False)
    return float(np.max(np.abs(eigenvalues)))
