# Projekt-Setup: HDC-Reservoir Experiment

## Auftrag
Lege ein neues Python-Projekt mit der unten beschriebenen Struktur an.
**Wichtig:** Implementiere KEINE Algorithmen. Nur Gerüst, leere Module mit
Docstrings, Konfigurationsdateien. Der Inhalt wird später vom Entwickler
selbst geschrieben.

## Projektname
`hdc-reservoir-mackey-glass`

## Ordnerstruktur

```
hdc-reservoir-mackey-glass/
├── README.md
├── requirements.txt
├── .gitignore
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── reservoir/
│   │   ├── __init__.py
│   │   ├── esn.py              # Echo State Network Klasse
│   │   └── dynamics.py         # Hilfsfunktionen für Reservoir-Dynamik
│   ├── hdc/
│   │   ├── __init__.py
│   │   ├── level_vectors.py    # Level-Hypervektor-Generierung
│   │   ├── operations.py       # Binding, Bundling, Cleanup
│   │   └── readout.py          # HDC-Readout-Klasse
│   ├── data/
│   │   ├── __init__.py
│   │   └── mackey_glass.py     # Mackey-Glass Zeitreihen-Generator
│   └── baselines/
│       ├── __init__.py
│       └── ridge_readout.py    # Ridge Regression Baseline
├── notebooks/
│   ├── 01_mackey_glass_exploration.ipynb
│   ├── 02_esn_ridge_baseline.ipynb
│   └── 03_esn_hdc_experiment.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_reservoir.py
│   ├── test_hdc.py
│   └── test_data.py
└── docs/
    └── design_notes.md         # Leere Datei für spätere Konzeptnotizen
```

## Inhalt der Dateien

### `README.md`
Erstelle eine README mit folgenden Abschnitten:
- **Projekttitel und kurze Beschreibung** (1-2 Sätze): Vergleich von Ridge
  Regression und Hyperdimensional Computing als Readout-Schicht für ein
  Echo State Network auf der Mackey-Glass-Zeitreihe.
- **Status:** Work in Progress – Lern- und Explorationsprojekt
- **Ziel:** Empirischer Vergleich der Effizienz und Genauigkeit beider
  Readout-Ansätze
- **Setup:** Platzhalter mit `pip install -r requirements.txt`
- **Struktur:** Kurze Erklärung der Ordner
- **Lizenz:** MIT (Platzhalter)

### `requirements.txt`
Folgende Pakete in aktuellen stabilen Versionen:
- numpy
- scipy
- torch
- matplotlib
- jupyter
- pytest
- ipykernel

### `.gitignore`
Standard Python `.gitignore` (von gitignore.io oder Standard GitHub Template).
Inklusive: `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `venv/`, `.venv/`,
`.vscode/`, `.idea/`, `*.egg-info/`, `dist/`, `build/`.

### `pyproject.toml`
Minimale Konfiguration für ein src-Layout-Projekt mit setuptools.
Python-Version >= 3.10.

### Modul-Dateien (alle `.py` in `src/`)
Jede Datei enthält **nur**:
1. Einen Modul-Docstring, der beschreibt, was das Modul später enthalten soll
2. Die Klassen-/Funktions-Stubs mit Docstrings, aber `pass` als Body oder
   `raise NotImplementedError`
3. Type Hints in den Signaturen, wo sinnvoll

#### Beispiel für `src/reservoir/esn.py`:
```python
"""Echo State Network Implementierung.

Dieses Modul enthält die Klasse für ein klassisches Echo State Network
mit tanh-Aktivierung und sparser, zufälliger Konnektivität.
"""

import numpy as np


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
        """Initialisiert das Reservoir.

        Args:
            n_reservoir: Anzahl der Neuronen im Reservoir.
            spectral_radius: Spektralradius der Reservoir-Gewichtsmatrix.
            sparsity: Anteil der nicht-null Verbindungen (0 bis 1).
            input_scaling: Skalierungsfaktor für Eingabegewichte.
            seed: Zufallsseed für Reproduzierbarkeit.
        """
        raise NotImplementedError

    def step(self, input_value: float) -> np.ndarray:
        """Führt einen Zeitschritt aus und gibt den Reservoir-Zustand zurück."""
        raise NotImplementedError

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Verarbeitet eine Eingabesequenz und gibt alle Zustände zurück."""
        raise NotImplementedError
```

Wende dieses Muster auf alle Module sinngemäß an. Die genauen Klassen- und
Funktionssignaturen wählst du sinnvoll basierend auf den Dateinamen und
Docstrings oben.

### Notebook-Dateien
Erstelle leere Jupyter-Notebooks mit jeweils:
- Einer Markdown-Zelle als Titel
- Einer Markdown-Zelle mit kurzer Zielbeschreibung
- Einer leeren Code-Zelle für Imports

### Test-Dateien
Leere pytest-Dateien mit Modul-Docstring und einem Platzhalter-Test,
der `pytest.skip("Noch nicht implementiert")` aufruft.

### `docs/design_notes.md`
Komplett leer bis auf einen Titel `# Design Notes`.

## Was du NICHT tun sollst

- Keine Algorithmus-Implementierungen schreiben
- Keine Mackey-Glass-Generierung implementieren
- Keine Tests mit echtem Inhalt schreiben
- Keine Beispieldaten generieren
- Keine zusätzlichen Bibliotheken hinzufügen, die nicht in
  requirements.txt stehen
- Keine virtuellen Umgebungen anlegen (mache ich selbst)
- Kein `git init` oder ähnliches (mache ich selbst)

## Abschluss

Wenn fertig, gib mir eine kurze Zusammenfassung zurück, was angelegt wurde,
und ob du auf Unklarheiten gestoßen bist, bei denen du eine Annahme treffen
musstest. Liste diese Annahmen explizit auf, damit ich sie nachträglich
prüfen kann.
