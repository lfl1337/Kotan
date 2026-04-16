"""Ridge Regression Readout-Baseline.

Dieses Modul enthält eine Ridge-Regression-Readout-Schicht als
klassische Baseline für den Vergleich mit dem HDC-Readout.
Ridge Regression ist der Standard-Readout für Echo State Networks.
"""

import numpy as np
from scipy import linalg


class RidgeReadout:
    """Ridge Regression Readout für Reservoir Computing.

    Klassischer linearer Readout mit L2-Regularisierung.
    Wird analytisch gelöst (kein iteratives Training nötig).
    """

    def __init__(self, alpha: float = 1e-6) -> None:
        self.alpha = alpha

    def fit(self, states: np.ndarray, targets: np.ndarray) -> None:
        """Trainiert den Readout analytisch: W = (X^T X + alpha I)^{-1} X^T y"""
        n_features = states.shape[1]
        A = states.T @ states + self.alpha * np.eye(n_features)
        b = states.T @ targets
        # linalg.solve ist numerisch stabiler als explizite Invertierung
        self.W_out = linalg.solve(A, b)

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Vorhersagen: y_pred = X @ W_out"""
        return states @ self.W_out

    def score(self, states: np.ndarray, targets: np.ndarray) -> float:
        """NRMSE = sqrt(mean((y_pred - y)^2)) / std(y)"""
        predictions = self.predict(states)
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        return float(rmse / np.std(targets))
