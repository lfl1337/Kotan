"""Mackey-Glass Zeitreihen-Generator.

Dieses Modul enthält Funktionen zur Erzeugung der Mackey-Glass
Differentialgleichung — ein chaotisches System, das als Standard-Benchmark
für Zeitreihenvorhersage mit Reservoir Computing verwendet wird.

Die Mackey-Glass-Gleichung:
    dx/dt = beta * x(t - tau) / (1 + x(t - tau)^n) - gamma * x(t)

Typische Parameter: beta=0.2, gamma=0.1, tau=17, n=10
"""

import numpy as np


def generate_mackey_glass(
    n_steps: int = 10000,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: float = 10.0,
    dt: float = 1.0,
    initial_value: float = 1.2,
    warmup: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """Generiert eine Mackey-Glass-Zeitreihe.

    Args:
        n_steps: Anzahl der zu generierenden Zeitschritte (nach Warmup).
        tau: Verzögerungsparameter (bestimmt Chaotizität).
        beta: Kopplungsparameter.
        gamma: Dämpfungsparameter.
        n: Exponent im Nenner.
        dt: Zeitschrittweite.
        initial_value: Anfangswert der Zeitreihe.
        warmup: Anzahl der Warmup-Schritte (werden verworfen).
        seed: Zufallsseed für eventuelle Rauschkomponente.

    Returns:
        1D-Array der Mackey-Glass-Zeitreihe (Länge n_steps).
    """
    laenge = tau + warmup + n_steps
    x = np.ones(laenge) * initial_value

    for t in range(tau, laenge - 1):
        x_alt = x[t - tau]
        x_jetzt = x[t]

        zufluss = beta * x_alt / (1 + x_alt**n)
        abfluss = gamma * x_jetzt
        veraenderung = zufluss - abfluss

        x[t + 1] = x_jetzt + dt * veraenderung

    return x[-n_steps:]


def prepare_dataset(
    timeseries: np.ndarray,
    prediction_horizon: int = 1,
    train_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Teilt die Zeitreihe in Trainings- und Testdaten.

    Args:
        timeseries: 1D-Array der Zeitreihe.
        prediction_horizon: Vorhersagehorizont (Schritte in die Zukunft).
        train_ratio: Anteil der Trainingsdaten (0 bis 1).

    Returns:
        Tuple aus (X_train, y_train, X_test, y_test).
    """
    X = timeseries[:-prediction_horizon]
    y = timeseries[prediction_horizon:]

    trenn = int(len(X) * train_ratio)
    X_train = X[:trenn]
    y_train = y[:trenn]
    X_test = X[trenn:]
    y_test = y[trenn:]

    return X_train, y_train, X_test, y_test