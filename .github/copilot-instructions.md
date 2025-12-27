Project: LAD-FLEX analysis (DCFlex)

Summary
- Purpose: Perform LAD-FLEX flexibility analysis on datacenter task traces, generate summary joblib exports and publication-ready plots.
- Data in: `DATA/` (CSV files like `df_aa_full_sorted_power.csv`, `df_tasks_zeta_070.csv`).
- Intermediate artifacts: `EXPORTS/df_summary_{SIM}_{SCENARIO}_{ZETA}.joblib`, `EXPORTS/res_dfs_*.joblib`, `EXPORTS/points_dates_*.joblib`.
- Plots: `PLOTS/` (subfolders per scenario and `SCENARIO_COMPARISON`).

Key entry-points & flow
- `main_export.py`: primary pipeline — reads `DATA/*`, runs LAD-FLEX via `utils.py` and `get_parameters.py`, writes `EXPORTS/*.joblib`. To produce exports run `python main_export.py` (edit top-level `ZETA`, `SIMULATION`, `SCENARIO`, or refactor to accept CLI args).
- `compare_scenarios.py`: reads `EXPORTS/df_summary_{SIM}_{SCENARIO}_{ZETA}.joblib` and `EXPORTS/points_dates_*` and produces figures in `PLOTS/SCENARIO_COMPARISON`. Run with `python compare_scenarios.py` after exports exist.
- Communication between components is file-based (CSV → joblib → PNG/CSV). Ensure matching `{SIMULATION}_{SCENARIO}_{zeta_suffix}` naming.

Important modules & responsibilities
- `utils.py`: core algorithms and helpers
  - `LAD_flex(...)` — selects deferrable tasks using latency, beta (overhang), delta (notification), gamma (buffer).
  - `fast_power_energy_series(...)` — builds hourly power/energy series from task intervals.
  - `apply_duration_uncertainty(...)` — perturbs runtimes for uncertainty experiments (zeta scenarios).
- `get_parameters.py`: centralized parameter sets returned by `return_params(simulation, scenario)`; used by `main_export.py` to enumerate latencies, windows, deltas, betas, gamma_buffers.

Conventions and patterns agents must follow
- Scenario names are uppercase strings: `BASELINE`, `DAM`, `mFRR`.
- ZETA naming: `ZETA_0`, `ZETA_010`, `ZETA_020`, ... used to pick input CSV and export filenames.
- Exports use joblib files named: `EXPORTS/df_summary_{SIMULATION}_{SCENARIO}_{zeta_suffix}.joblib` — plotting scripts expect that exact pattern.
- Time indices in `df_summary` are a multi-index: `['timestamp','latency','flex_window','delta_notification','beta','gamma_buffer']`.

Developer workflows & gotchas
- To regenerate analysis for a scenario: run `main_export.py` with matching `ZETA`, `SIMULATION`, `SCENARIO` then run `compare_scenarios.py` to produce figures.
- `main_export.py` reads large CSVs; use the `N_slice` and `N_slice_start` variables near the top to limit rows when developing locally.
- `main_export.py` currently sets parameters via top-file variables (not CLI). Preferred quick task: convert those to `argparse` flags so agents can run CI-friendly commands.
- File-not-found handling: plotting code assumes joblib exists; add explicit guards and helpful error messages when joblib files are missing.

Concrete tasks an AI agent can do (examples)
- Add CLI for `main_export.py` to accept `--zeta`, `--simulation`, `--scenario`, and `--rows`.
- Add unit smoke-tests that run a tiny subset (use `N_slice=1000`) and assert `EXPORTS/*.joblib` are created.
- Harden `compare_scenarios.py` to check for `EXPORTS/df_summary_*` and fall back to a short message with which file is missing.
- Add `requirements.txt` listing `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`.

Dependencies (discoverable from files)
- pandas, numpy, matplotlib, seaborn, joblib

Where to look for examples
- Plot generation patterns: `compare_scenarios.py` (boxplots, bar charts, time-series overlays, normalized comparisons).
- Orchestration & parameterization: `main_export.py` + `get_parameters.py`.
- Core algorithm and data transforms: `utils.py`.

If anything is unclear or you want the agent to implement one of the concrete tasks above (CLI, tests, guards, requirements), tell me which task to prioritize and I will proceed.
