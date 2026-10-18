# Task B — Neural Network from Scratch (pure NumPy)

`npnn/` is the **deliverable**: forward + backward for Linear, GELU, CrossEntropyLoss, Adam,
Conv2d and SwitchNorm2d, plus a CIFAR-100 data pipeline. It imports **numpy only** —
enforced by `tests/test_purity.py` (AST scan + clean-subprocess `sys.modules` check).

**Requirements:** the shared project environment (numpy, matplotlib, pandas — already installed;
nothing new is installed for Task B). Data is read **read-only** from `../data/` (CIFAR-100 pickles).

**How to run** (from `task_b/`):
- `python tests/test_purity.py`  — purity guard
- `python tests/test_modules.py` — per-module parity vs PyTorch (torch allowed here: it is tooling)
- `jupyter lab notebooks/`       — 10 → 13; every notebook runs top-to-bottom

## Deliverable vs. verification tooling

| Path | Role | May import |
|---|---|---|
| `npnn/` | **deliverable implementation** | numpy only |
| `tests/`, `notebooks/` | verification / comparison tooling | torch, matplotlib, … |
| `runs/` | experiment archives (`config.json`, `metrics.csv`, `curves.png`, `params.npz`) | — |
| `report/` | report sources (merged with Part A into one PDF) | — |

Part A of this assignment lives in `../src`, `../notebooks`, `../runs`, `../data`;
`task_b/` never writes to any of them.
