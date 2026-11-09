# Learning the Equilibrium Operator

Code, benchmark, trained weights and raw results for the paper

> F. Khojasteh, *Learning the Equilibrium Operator: Amortized Fixed Point Analysis with
> Physics-Informed Neural Networks, Calibrated Uncertainty, and A Posteriori Certificates*,
> submitted to *Expert Systems with Applications*.

A physics-informed mixture density network maps a sampled scalar map `f` to all of its fixed
points, their stability labels and multipliers in one forward pass. Split conformal calibration
gives simultaneous coverage of all fixed points, and interval-arithmetic Banach/Krawczyk tests
certify individual predictions.

## Requirements

Python 3.11, CPU only:

```
pip install -r requirements.txt
```

## Files

| File | Purpose |
|---|---|
| `functions.py` | Function families, verified ground truth, interval evaluation |
| `gen_data.py` | Generates the benchmark and the zero-shot suites (`data/benchmark.pkl`) |
| `model.py` | Mixture density network and losses |
| `train.py` | Trains the main model and the ablations |
| `cert1d.py` | Banach- and Krawczyk-type certificates in interval arithmetic |
| `evaluate.py` | Portraits, calibration, conformal, certificates, baselines, bifurcation sweep |
| `eval_margin.py` | Control experiment for the classical dynamical systems |
| `retime.py` | Re-measures wall-clock timings |
| `make_results.py` | Writes every table, figure and number of the paper |
| `vizstyle.py` | Figure style |
| `data/` | Benchmark |
| `results/` | Trained weights (`*.pt`), training histories and evaluation results |

## Reproducing the paper

All commands are run from this directory.

```
python gen_data.py                       # benchmark (already included in data/)
python train.py full                     # main model
python train.py conv --encoder conv      # ablations
python train.py lstm --encoder lstm
python train.py K1 --K 1
python train.py point --point
python train.py nophys --nophys
python evaluate.py                       # evaluation of all models
python eval_margin.py
python make_results.py                   # tables, figures and macros
```

The saved weights and results are included, so `python make_results.py` alone regenerates all
tables, figures and numbers from them. It writes `tables/`, `figures/` and `results_macros.tex`
into the parent directory, where the LaTeX source of the paper is expected.
