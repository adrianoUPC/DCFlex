# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DCFlex implements the **LAD-FLEX** (Latency-Aware Deferrable Flexibility) algorithm for analyzing datacenter task-level energy flexibility. It identifies deferrable computing tasks based on their wait-time latency and computes how much energy can be shifted within flexibility windows — relevant for grid services like mFRR, day-ahead markets, and baseline scenarios.

The codebase is a research analysis pipeline (not a library/package). All scripts are run as standalone Python files from the project root.

## Running the Pipeline

```bash
# 1. Generate exports (edit ZETA, SIMULATION, SCENARIO at top of file before running)
python main_export.py

# 2. Generate per-scenario plots (edit SIMULATION, SCENARIO, ZETA at top of import_DEFAULT.py)
python import_DEFAULT.py

# 3. Compare scenarios (requires all 3 scenario exports to exist first)
python compare_scenarios.py

# 4. Compare uncertainty (zeta) impact on deferability classification
python compare_uncertainty.py

# 5. Generate perturbed task CSVs (run from uncertainty/ subfolder)
cd uncertainty && python task_duration_uncertainty.py
```

There are no tests, no CLI arguments, no build system, and no `requirements.txt`. Parameters are set by editing variables at the top of each script.

## Dependencies

pandas, numpy, matplotlib, seaborn, joblib

## Architecture and Data Flow

```
DATA/*.csv  -->  main_export.py  -->  EXPORTS/*.joblib  -->  import_DEFAULT.py      --> PLOTS/{SIM}_{SCENARIO}_{ZETA}/
                                                         -->  compare_scenarios.py   --> PLOTS/SCENARIO_COMPARISON/
                                                         -->  compare_uncertainty.py --> PLOTS/ + EXPORTS/metrics_*.csv
```

### Core Modules

- **`utils.py`** — Core algorithm. `LAD_flex()` selects deferrable tasks using 5 parameters: latency threshold, flex window [t1,t2], beta (overhang tolerance), delta (notification period), gamma (buffer). Also contains `fast_power_energy_series()` for building hourly power/energy time series from task intervals, and `apply_duration_uncertainty()` for zeta perturbation experiments.
- **`get_parameters.py`** — `return_params(simulation, scenario)` returns parameter dictionaries. Two simulation modes: `DEFAULT` (single parameter set per scenario) and `SENSITIVE` (parameter sweeps).
- **`utils_plotting.py`** — `plot_ladflex_before_after()` for before/after LAD-Flex power visualization.
- **`uncertainty/utils_uncertainty.py`** — Duplicate of `apply_duration_uncertainty` from `utils.py` (uses `zeta` parameter name instead of `gamma`).

### Key Scripts

- **`main_export.py`** — Primary pipeline: loads task CSVs, runs LAD-FLEX across all parameter combos x hourly timestamps, exports `df_summary` and `res_dfs` as joblib. Use `N_slice`/`N_slice_start` variables to limit rows during development.
- **`import_DEFAULT.py`** — Loads exports, generates per-scenario plots: deferrable energy time series, 3x3 power grids for 9 notable points (P1-P3 top, Q1-Q3 at 75th percentile, R1-R3 at median), KDE, before/after, and summary tables.
- **`compare_scenarios.py`** — Cross-scenario comparison: boxplots, time series overlays, notable points bar charts, hourly efficiency (kWh/h). Expects all 3 scenario exports to exist.
- **`compare_uncertainty.py`** — Builds confusion matrices comparing deferability decisions at different zeta values vs baseline (zeta=0). Computes accuracy, precision, recall, F1.

## Key Conventions

- **Scenario names**: `BASELINE`, `DAM`, `mFRR` (uppercase). In plots, DAM is displayed as `INTRADAY`.
- **ZETA naming**: `ZETA_0`, `ZETA_010`, `ZETA_020`, `ZETA_030`, `ZETA_050`, `ZETA_070` — used in both input CSV filenames and export filenames.
- **Export naming**: `EXPORTS/df_summary_{SIMULATION}_{SCENARIO}_{zeta_suffix}.joblib` and `EXPORTS/res_dfs_*.joblib`, `EXPORTS/points_dates_*.joblib`. Plotting scripts expect these exact patterns.
- **df_summary multi-index**: `['timestamp', 'latency', 'flex_window', 'delta_notification', 'beta', 'gamma_buffer']`.
- **res_dfs dict keys**: `(date, latency, flex_window, delta_notification, beta, gamma_buffer)` tuples.
- **Sample week**: `1970-02-01` to `1970-02-07` is used as the representative analysis period across scripts.
- **Colorblind-friendly palette**: BASELINE=#0571B0 (blue), DAM=#FC9272 (red), mFRR=#A6D854 (green).
- **Plot style**: Publication-ready with large fonts (16-20pt labels, 14-16pt ticks), seaborn whitegrid, visible dark spines, DPI 300. Most plots save both PNG and PDF.
