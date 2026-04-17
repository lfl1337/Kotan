# Design Notes

---

## 2026-04-17 — Experiment 01: ESN + Ridge Regression Baseline

### Was gebaut wurde

- `src/reservoir/dynamics.py` — Sparse-Matrix-Generierung, Spektralradius-Berechnung und -Skalierung via `scipy.sparse.linalg.eigs`
- `src/reservoir/esn.py` — `EchoStateNetwork` mit tanh-Aktivierung, `scipy.sparse` Reservoir-Matrix, deterministischem Seed
- `src/baselines/ridge_readout.py` — `RidgeReadout` mit analytischer Lösung via `scipy.linalg.solve`, NRMSE-Score
- `tests/test_data.py`, `tests/test_reservoir.py`, `tests/test_ridge.py` — 22 Tests, alle grün
- `notebooks/02_esn_ridge_baseline.ipynb` — Vollständiges Experiment-Notebook mit Alpha-Vergleich

### Alpha-Vergleich (Hauptbefund)

| alpha | Train-NRMSE | Test-NRMSE | Bewertung |
|-------|-------------|------------|-----------|
| 1e-6  | 0.0007 | 1.1475 | Katastrophales Overfitting |
| 1e-4  | 0.0015 | 0.5496 | Noch stark überangepasst |
| 1e-2  | 0.0040 | 0.2735 | Besser, aber > 0.1 |
| 10    | 0.0250 | 0.0914 | Zielbereich (< 0.1) erreicht |
| 100   | 0.0491 | 0.0721 | Gutes Ergebnis |

### Lernbefund: Gram-Matrix-Konditionierung

Die Gram-Matrix X^T X des Reservoirs (500 Neuronen, 9499 Trainingspunkte) hat Konditionszahl ~2×10^11.
Eigenwerte liegen zwischen 1×10^-6 und 2×10^5.

**Warum alpha=1e-6 versagt:** Ridge-Regularisierung addiert alpha auf die Diagonale der Gram-Matrix. Bei alpha=1e-6 ist die Regularisierung im Vergleich zu den dominanten Eigenwerten (~10^5) vollständig unsichtbar. Das Modell invertiert die ill-conditioned Matrix ohne effektive Dämpfung — es memoriert die Trainingsdaten perfekt (NRMSE=0.0007) und generalisiert nicht (NRMSE=1.15).

**Faustregel:** Für ESN + Ridge auf Mackey-Glass mit ~500 Neuronen und ~10k Trainingspunkten: alpha ≥ 1, typisch 10–100.

### Architektur-Entscheidungen

- `scipy.sparse.csr_matrix` für W_reservoir — Speichereffizienz bei 10% Dichte (25k statt 250k Einträge)
- `scipy.linalg.solve` statt expliziter Invertierung — numerisch stabiler bei schlecht konditionierten Matrizen
- `scipy.sparse.linalg.eigs(k=1)` für Spektralradius — O(n) statt O(n²)
- `v0=np.ones(n)` als ARPACK-Startvektor — macht `eigs()` deterministisch (kein floating-point-Nichtdeterminismus)
- Warmup 100 Schritte: Einschwingzeit des Reservoirs wird nicht fürs Training verwendet

### Offene Fragen

- NRMSE von 0.07–0.09 ist unter der Spec-Schwelle 0.1, aber über dem "typischen" Bereich 0.01–0.05
  → Möglicher Grund: Reset des Reservoirs zwischen Train und Test (Einschwingzeit der Test-Sequenz)
  → Alternativ: kein Reset + Test-Warmup könnte bessere Ergebnisse liefern
- Nächster Schritt: HDC-Readout (Level-Hypervektoren) als Alternative — Auftrag 02
