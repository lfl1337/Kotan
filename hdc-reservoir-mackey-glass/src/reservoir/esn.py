"""Echo State Network Implementierung.

Dieses Modul enthält die Klasse für ein klassisches Echo State Network
mit tanh-Aktivierung und sparser, zufälliger Konnektivität.
"""

import numpy as np
from scipy import sparse

from src.reservoir.dynamics import generate_sparse_matrix, scale_spectral_radius


class EchoStateNetwork:
    """Echo State Network mit konfigurierbarer Reservoir-Größe und Spektralradius."""

    def __init__(
        self,
        n_reservoir: int = 500,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        input_scaling: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.n_reservoir = n_reservoir
        rng = np.random.default_rng(seed)
        # Separater seed für W_input (reproduzierbar, unabhängig von W_reservoir)
        seed_input = int(rng.integers(0, 2**31))
        seed_reservoir = int(rng.integers(0, 2**31))

        W_raw = generate_sparse_matrix(n_reservoir, sparsity, seed=seed_reservoir)
        self.W_reservoir = scale_spectral_radius(W_raw, spectral_radius)

        rng_input = np.random.default_rng(seed_input)
        self.W_input = rng_input.uniform(-1, 1, n_reservoir) * input_scaling

        self.state = np.zeros(n_reservoir)

    def step(self, input_value: float) -> np.ndarray:
        """Ein Zeitschritt: x(t+1) = tanh(W_input * u + W_reservoir @ x(t))"""
        pre_activation = self.W_input * input_value + self.W_reservoir @ self.state
        if sparse.issparse(pre_activation):
            pre_activation = pre_activation.toarray().flatten()
        self.state = np.tanh(pre_activation)
        return self.state.copy()

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Verarbeitet eine Eingabesequenz, gibt alle Zustände zurück (T x n_reservoir)."""
        states = np.zeros((len(inputs), self.n_reservoir))
        for t, u in enumerate(inputs):
            states[t] = self.step(u)
        return states

    def reset(self) -> None:
        """Setzt den internen Zustand auf Null zurück."""
        self.state = np.zeros(self.n_reservoir)
