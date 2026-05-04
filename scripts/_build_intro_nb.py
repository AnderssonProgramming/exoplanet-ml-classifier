"""One-off helper: rebuild notebooks/00_intro.ipynb from clean source.

This script is committed for reproducibility but is *not* part of the runtime
pipeline. Run it manually if you want to regenerate the introduction
notebook.
"""

from pathlib import Path

import nbformat as nbf

INTRO_MD_TITLE = """\
# 00 — Project Introduction
**exoplanet-ml-classifier** | MLEA_M — ECI 2026-1

Welcome. This first notebook does not run any model — its only job is to set the
scientific stage and remind the reader **why** the rest of the pipeline exists.

---
## Mission context — Kepler & the KOI table

NASA's **Kepler Space Telescope** stared at the same patch of sky for four years
and recorded brightness time-series for ~150 000 stars. Whenever a planet
crosses in front of its host star (a *transit*), the star dims by a tiny,
periodic amount. Kepler's data-reduction pipeline flags each candidate event
as a **Kepler Object of Interest (KOI)**.

Not every KOI is a planet. Many are **eclipsing binary stars**, **stellar
variability**, or **instrument noise**. The KOI table records, for each
candidate:

* photometric transit parameters (period, depth, duration, SNR, ...),
* stellar properties of the host (effective temperature, radius, surface
  gravity, ...),
* and a final **disposition** decided by the NASA vetting team.
"""

INTRO_MD_PROBLEM = """\
## What problem are we solving?

> **Given the photometric and stellar features of a KOI, decide whether it is
> a genuine planet candidate or a false positive — without using any column
> that was produced *after* the vetting decision was made.**

This is **supervised binary classification** with a moderately imbalanced
target (~49 % CANDIDATE / ~51 % FALSE POSITIVE) on tabular data.

We deliberately do **not** use:
* `koi_disposition` — the label itself,
* `koi_score` — NASA's own probability,
* `koi_fpflag_*` — false-positive flags raised during vetting.

Including any of those would give a model that scores ~99 % and predicts
nothing useful for new candidates.
"""

INTRO_MD_IMPACT = """\
## Why this matters

| Stakeholder | Benefit |
|---|---|
| **NASA Exoplanet Archive operators** | Faster triage of incoming KOIs before manual review. |
| **Astronomers** | Higher-purity follow-up list → fewer wasted nights on the telescope. |
| **The public / education** | An open, explainable baseline anyone can reproduce on a laptop. |

Reducing false negatives (missed planets) is more important than reducing
false positives (wasted follow-up). That is why we report **F1**, **MCC**, and
do **threshold tuning** instead of optimising plain accuracy.
"""

INTRO_MD_HERITAGE = """\
## NASA Space Apps Challenge 2025 — heritage

This project continues the *Exoplanet Hunter AI* prototype built by team
**ECI Centauri** at the **NASA Space Apps Challenge 2025**, which earned
a **Global Finalist** placement.

| Component | Repository |
|---|---|
| ⚛️ React + Three.js front-end | <https://github.com/JAPV-X2612/ECI-Centauri-Frontend> |
| ⚙️ FastAPI back-end            | <https://github.com/JAPV-X2612/ECI-Centauri-Backend>  |
| 🧠 Astronet light-curve CNN    | <https://github.com/Ch0comilo/astronet-cnn-v3>        |

The hackathon system shipped a working web app + CNN backend in 48 hours.
This repository tackles the **same scientific question** with course-grade
ML methodology, full reproducibility, and a rigorous evaluation protocol.
"""

INTRO_MD_GUIDE = """\
## How to read the rest of this work

| Notebook | Question it answers |
|---|---|
| `01_eda.ipynb` | What does the data look like and what should we worry about? |
| `02_preprocessing.ipynb` | How do we clean it without leaking the label? |
| `03_models_and_metrics.ipynb` | Which of five classifier families wins, and by how much? |
| `04_final_presentation.ipynb` | One executable run that reproduces everything end-to-end. |

A full IEEE-format paper accompanies the code in `reports/paper/`, and the
class-presentation slides in `reports/slides/`.
"""

INTRO_MD_AUTHORS = """\
## Authors & course

| Role | Name |
|---|---|
| Student | **Andersson David Sánchez Méndez** |
| Student | **Cristian Santiago Pedraza Rodríguez** |
| Professor | **Mario Julián Cañón Ayala** |
| Professor | **Hector Javier Hortua Orjuela** |

🎓 *MLEA_M — Machine Learning · Maestría en Ciencia de Datos*
🏫 *Escuela Colombiana de Ingeniería Julio Garavito · 2026-1*
"""

INTRO_CODE = """\
import sys
from pathlib import Path

ROOT = Path().resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (  # noqa: F401  -- import-only sanity check
    constants,
    data_loader,
    evaluation,
    models,
    preprocessing,
    visualization,
)

print(f"Python      : {sys.version.split()[0]}")
print(f"Project root: {ROOT}")
print(f"Random seed : {constants.RANDOM_SEED}")
print("All src/ modules imported successfully.")
"""


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(INTRO_MD_TITLE),
        nbf.v4.new_markdown_cell(INTRO_MD_PROBLEM),
        nbf.v4.new_markdown_cell(INTRO_MD_IMPACT),
        nbf.v4.new_markdown_cell(INTRO_MD_HERITAGE),
        nbf.v4.new_markdown_cell(INTRO_MD_GUIDE),
        nbf.v4.new_markdown_cell(INTRO_MD_AUTHORS),
        nbf.v4.new_markdown_cell("## Sanity check — environment ready?"),
        nbf.v4.new_code_cell(INTRO_CODE),
    ]
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }

    out_path = Path(__file__).resolve().parent.parent / "notebooks" / "00_intro.ipynb"
    nbf.write(nb, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
