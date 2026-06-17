# PROJECT-SHOWCASE — LAD-FLEX datacenter flexibility, end to end

A self-contained walkthrough of the **LAD-FLEX** (Latency-Aware Deferrable
Flexibility) method for quantifying task-level energy flexibility in a
datacenter. It tells the whole story as runnable Jupyter notebooks:

**explore the data → define the method → run all computations → test robustness.**

This folder is fully standalone: it ships its own data sample, defines every
function in one module (`lad_flex.py`), and reads/writes only within this folder.

## Contents

```
PROJECT-SHOWCASE/
├── README.md
├── lad_flex.py                        # all project functions (single import surface)
├── data/
│   └── df_tasks_sample.csv            # 100k-task sample (10 columns, ~9 MB)
├── outputs/                           # figures, tables & cached results (written by notebooks)
├── 01_exploratory_data_analysis.ipynb
├── 02_lad_flex_method.ipynb
├── 03_scenario_comparison.ipynb
└── 04_uncertainty_sensitivity.ipynb
```

## Run order

Run the notebooks top to bottom in order:

1. **`01_exploratory_data_analysis.ipynb`** — dataset overview, distributions,
   workload composition, total DC load over the sample week, and how much load is
   latency-tolerant.
2. **`02_lad_flex_method.ipynb`** — walks through the `LAD_flex` deferability
   logic on a single window, then runs the full hourly sweep across the three
   scenarios (**BASELINE**, **DAM** = *Intraday*, **mFRR**). Produces the
   deferrable-energy time series, the 3×3 notable-window grid, the before/after
   load-reshaping plot, and a breakdown table. **Caches per-scenario results to
   `outputs/` for notebooks 03 and 04.**
3. **`03_scenario_comparison.ipynb`** — cross-scenario boxplots, time-series
   overlay, hourly flexibility efficiency, and a summary table. Reuses the cached
   results (recomputes automatically if they are missing).
4. **`04_uncertainty_sensitivity.ipynb`** — duration-prediction uncertainty (ζ):
   error distributions, confusion matrices vs the ζ=0 baseline, and
   accuracy/precision/recall/F1 trends; plus a LAD-FLEX parameter sensitivity
   sweep. All perturbations are generated in-notebook — no extra data files.

Notebook 02 must be run before 03/04 for the fast path; otherwise 03/04 recompute
what they need.

## Data provenance

`data/df_tasks_sample.csv` is a 100,000-task slice of a sorted GPU-cluster trace.
It spans **1970-01-26 → 1970-02-08** and contains the full **sample week
1970-02-01 → 1970-02-07** that all plots focus on.

Columns: `task_name, start_time, end_time, plan_gpu, plan_mem, gpu_type,
plan_cpu, wait_time, runtime_i, plan_power`. The six core columns
(`task_name, start_time, end_time, wait_time, runtime_i, plan_power`) drive the
algorithm; the GPU/CPU/memory columns enrich the EDA.

## The method in one paragraph

For a flexibility window `[t1, t2]`, a task is **deferrable** when it is
latency-tolerant (`wait_time ≥ λ̂`), fits inside the whiskered window
`[t0, tend]` with `t0 = t1 − δ` (notification) and `tend = t2 + γ` (buffer), and
does not overhang `[t1, t2]` by more than a fraction `β` of its runtime on either
side. See `LAD_flex` and `run_lad_flex_sweep` in `lad_flex.py`.

## Scenarios & palette

| Scenario | Service | Colour |
|---|---|---|
| BASELINE | energy shifting | `#0571B0` (blue) |
| DAM (*Intraday*) | day-ahead / intraday | `#FC9272` (red) |
| mFRR | manual frequency restoration reserve | `#A6D854` (green) |

## Dependencies

`pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`, and Jupyter. No build
system or CLI — just open the notebooks.

## Note on performance

`fast_power_energy_series` in `lad_flex.py` uses a vectorized sweep-line
(events + cumulative sum) that runs in O(n log n) instead of O(segments × tasks).
The global power series over 100k tasks computes in well under a second, so the
notebooks run interactively.
