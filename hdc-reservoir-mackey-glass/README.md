# HDC-Reservoir Mackey-Glass

Vergleich von Ridge Regression und Hyperdimensional Computing (HDC) als
Readout-Schicht für ein Echo State Network (ESN) auf der Mackey-Glass-Zeitreihe.

## Status

🚧 Work in Progress — Lern- und Explorationsprojekt

## Ziel

Empirischer Vergleich der Effizienz und Genauigkeit beider Readout-Ansätze:
- **Baseline:** Ridge Regression (klassisch, analytisch)
- **Experiment:** HDC-basiertes Readout (neuromorph-inspiriert)

## Setup

```bash
pip install -r requirements.txt
```

## Projektstruktur

| Ordner       | Beschreibung                                      |
|--------------|---------------------------------------------------|
| `src/reservoir/` | Echo State Network und Reservoir-Dynamik     |
| `src/hdc/`       | Hyperdimensional Computing Module            |
| `src/data/`      | Mackey-Glass Zeitreihen-Generator            |
| `src/baselines/` | Ridge Regression Baseline                    |
| `notebooks/`     | Jupyter-Notebooks für Exploration            |
| `tests/`         | pytest-Tests                                 |
| `docs/`          | Design-Notizen und Dokumentation             |

## Lizenz

Apache License 2.0
