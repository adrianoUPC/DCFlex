"""
lad_flex.py — LAD-FLEX (Latency-Aware Deferrable Flexibility) functions
========================================================================

Single import surface for the PROJECT-SHOWCASE notebooks. Consolidates the core
algorithm, the power/energy series builder, the parameter sets, the duration
uncertainty model, the before/after plotting helper, and one convenience wrapper
(`run_lad_flex_sweep`) that runs the full hourly sweep used to build `df_summary`
and `res_dfs`.

Everything the showcase needs lives in this module — the notebooks only import
from here.

Conventions:
  - df_summary multi-index:
      ['timestamp', 'latency', 'flex_window', 'delta_notification', 'beta', 'gamma_buffer']
  - res_dfs dict keys:
      (date, latency, flex_window, delta_notification, beta, gamma_buffer)
  - Scenario names: BASELINE, DAM, mFRR  (DAM is displayed as "INTRADAY" in plots)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Colour palette (colourblind-friendly) shared across the showcase notebooks
# ---------------------------------------------------------------------------
SCENARIO_COLORS = {
    "BASELINE": "#0571B0",   # blue
    "DAM": "#FC9272",        # red   (displayed as INTRADAY)
    "mFRR": "#A6D854",       # green
}
SCENARIO_DISPLAY = {"BASELINE": "Baseline", "DAM": "Intraday", "mFRR": "mFRR"}


# ---------------------------------------------------------------------------
# Power / energy time series
# ---------------------------------------------------------------------------
def get_timeline(df):
    """Sorted unique vector of all task start/end times (the breakpoints)."""
    return pd.concat([df['start_time'], df['end_time']]).sort_values().unique()


def build_task_intervals(df):
    """IntervalIndex [start_time, end_time) for every task."""
    return pd.IntervalIndex.from_arrays(df['start_time'], df['end_time'], closed='left')


def fast_power_energy_series(df, timeline):
    """
    Compute the power and energy step series over `timeline`.

    Returns a DataFrame indexed by time with columns 'power_kW' and 'energy_kWh'.
    Within each segment [timeline[i], timeline[i+1]) the active power is the sum of
    `plan_power` (W) of all tasks overlapping that segment (converted to kW), and
    energy is power times the segment duration in hours.

    Implementation: vectorized sweep-line. Each task contributes a +plan_power
    event at its start_time and a -plan_power event at its end_time; the running
    cumulative sum at a breakpoint is the power active in the segment that starts
    there. With task intervals closed on the left ([start, end)), a task ending
    exactly at a breakpoint is (correctly) inactive in the following segment.
    This is mathematically identical to the IntervalIndex-overlap formulation but
    runs in O(n log n) instead of O(segments x tasks).
    """
    if df is None or len(df) == 0 or timeline is None or len(timeline) < 2:
        return pd.DataFrame(columns=['power_kW', 'energy_kWh'],
                            index=pd.DatetimeIndex([], name='time'))

    starts = pd.to_datetime(df['start_time']).values.astype('datetime64[ns]')
    ends = pd.to_datetime(df['end_time']).values.astype('datetime64[ns]')
    power_kw = df['plan_power'].to_numpy(dtype=float) / 1000.0  # W -> kW

    event_times = np.concatenate([starts, ends])
    event_deltas = np.concatenate([power_kw, -power_kw])

    order = np.argsort(event_times, kind='mergesort')
    event_times = event_times[order]
    event_deltas = event_deltas[order]

    # Aggregate the deltas occurring at each distinct breakpoint, then accumulate.
    unique_times, first_idx = np.unique(event_times, return_index=True)
    summed = np.add.reduceat(event_deltas, first_idx)
    cum_power = np.cumsum(summed)  # power active in segment starting at unique_times[i]

    tl = np.asarray(timeline).astype('datetime64[ns]')
    seg_start = tl[:-1]
    seg_end = tl[1:]
    # unique_times == timeline (same set of breakpoints); align defensively.
    pos = np.searchsorted(unique_times, seg_start)
    power_series = cum_power[pos]
    duration_h = (seg_end - seg_start) / np.timedelta64(1, 'h')
    energy_series = power_series * duration_h

    df_res = pd.DataFrame({'time': seg_start,
                           'power_kW': power_series,
                           'energy_kWh': energy_series})
    df_res['time'] = pd.to_datetime(df_res['time'])
    df_res.set_index('time', inplace=True)
    return df_res


# ---------------------------------------------------------------------------
# Core LAD-FLEX algorithm
# ---------------------------------------------------------------------------
def LAD_flex(df_a, LATENCY, t1, t2, beta_overhang, delta_notification, gamma_buffer):
    """
    Select deferrable tasks for a single flexibility window.

    Parameters
    ----------
    df_a : pd.DataFrame
        Tasks with 'start_time', 'end_time', 'wait_time', 'runtime_i'.
    LATENCY : float
        Minimum wait_time (s) for a task to be considered latency-tolerant.
    t1, t2 : pd.Timestamp
        Start / end of the flexibility window.
    beta_overhang : float
        Overhang tolerance (fraction of runtime allowed to spill outside [t1,t2]).
    delta_notification : pd.Timedelta
        Notification period; whisker before t1 (t0 = t1 - delta_notification).
    gamma_buffer : pd.Timedelta
        Buffer after t2 (tend = t2 + gamma_buffer).

    Returns
    -------
    (df_deferrable, df_not_deferrable) : tuple of pd.DataFrame
    """
    # 1. Latency filter
    df_flex = df_a[df_a['wait_time'] >= LATENCY].copy()

    # 2. Whiskered window [t0, tend]
    t0 = t1 - delta_notification
    tend = t2 + gamma_buffer

    # 3. Keep only tasks overlapping the whiskered window (cheap pre-filter)
    task_intervals = pd.IntervalIndex.from_arrays(df_flex['start_time'], df_flex['end_time'], closed='left')
    window_interval = pd.Interval(left=t0, right=tend, closed='left')
    overlaps_mask = task_intervals.overlaps(window_interval)
    df_flex = df_flex[overlaps_mask].copy()

    # 4. Deferrability condition with beta-overhang allowance
    df_flex["Deferrable"] = (
        (df_flex['start_time'] >= t0) &
        (df_flex['end_time'] <= tend) &
        ((t1 - df_flex['start_time']).dt.total_seconds() <= beta_overhang * df_flex['runtime_i']) &
        ((df_flex['end_time'] - t2).dt.total_seconds() <= beta_overhang * df_flex['runtime_i'])
    )

    df_deferrable = df_flex[df_flex["Deferrable"]].copy()
    df_not_deferrable = df_flex[~df_flex["Deferrable"]].copy()
    return df_deferrable, df_not_deferrable


def Value(delta):
    """
    Flexibility value (EUR/kWh) as a function of notification period `delta`
    (pd.Timedelta). Shorter notification => more valuable grid service.

      - delta <= 5 min:   0.25   (real-time reserves)
      - 5 < delta <= 30:  0.12   (fast reserves, aFRR/mFRR)
      - 30 < delta <= 60: 0.06   (slow reserves)
      - delta > 60:       0.03   (energy shifting)
    """
    delta = delta.total_seconds() / 60  # minutes
    if delta <= 5:
        return 0.25
    elif delta <= 30:
        return 0.12
    elif delta <= 60:
        return 0.06
    else:
        return 0.03


# ---------------------------------------------------------------------------
# Duration uncertainty model
# ---------------------------------------------------------------------------
def apply_duration_uncertainty(df, zeta, random_seed=42):
    """
    Apply normally distributed prediction error to task durations.

    Predicted duration d_hat_k = d_k + eps_k, with eps_k ~ N(0, (zeta*d_k)^2),
    clipped to a 1 s minimum. `end_time` is recomputed from the predicted
    duration (and the wait_time), keeping the original as 'end_time_actual'.

    Parameters
    ----------
    df : pd.DataFrame
        Original tasks with 'runtime_i', 'start_time', 'end_time', 'wait_time'.
    zeta : float
        Relative prediction error (e.g. 0.2 for 20%).
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy with perturbed 'end_time' and diagnostic columns
        (duration_predicted, duration_actual, prediction_error,
        prediction_error_pct, end_time_actual).
    """
    np.random.seed(random_seed)
    df_perturbed = df.copy()

    d_actual = df_perturbed["runtime_i"].values
    sigma = zeta * d_actual
    epsilon = np.random.normal(loc=0, scale=sigma, size=len(d_actual))
    d_predicted = np.maximum(d_actual + epsilon, 1.0)

    df_perturbed["duration_predicted"] = d_predicted
    df_perturbed["duration_actual"] = d_actual
    df_perturbed["prediction_error"] = epsilon
    # guard against zero-runtime tasks when expressing the error as a percentage
    with np.errstate(divide="ignore", invalid="ignore"):
        df_perturbed["prediction_error_pct"] = np.where(
            d_actual != 0, (epsilon / d_actual) * 100, 0.0)

    df_perturbed["end_time_actual"] = df_perturbed["end_time"]
    df_perturbed["end_time"] = (
        df_perturbed["start_time"]
        + pd.to_timedelta(d_predicted, unit="s")
        + pd.to_timedelta(df_perturbed["wait_time"], unit="s")
    )
    return df_perturbed


# ---------------------------------------------------------------------------
# Parameter sets (DEFAULT single point per scenario; SENSITIVE sweeps)
# ---------------------------------------------------------------------------
def return_params(simulation, scenario):
    """Return the parameter dictionary for a given simulation mode / scenario."""
    if simulation == "DEFAULT":
        if scenario == "BASELINE":
            latencies = [600]
            flex_windows = [pd.Timedelta(hours=3)]
            delta_notifications = [pd.Timedelta(minutes=60)]
            betas = [0.7]
            gamma_buffers = [pd.Timedelta(minutes=60)]
        elif scenario == "DAM":
            latencies = [60 * 30]
            flex_windows = [pd.Timedelta(hours=4)]
            delta_notifications = [pd.Timedelta(minutes=60 * 3)]
            betas = [0.8]
            gamma_buffers = [pd.Timedelta(minutes=120)]
        elif scenario == "mFRR":
            latencies = [60 * 3]
            flex_windows = [pd.Timedelta(hours=1)]
            delta_notifications = [pd.Timedelta(minutes=15)]
            betas = [0.9]
            gamma_buffers = [pd.Timedelta(minutes=20)]
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    elif simulation == "SENSITIVE":
        if scenario == "BASELINE":
            latencies = [600, 900]
            flex_windows = [pd.Timedelta(hours=h) for h in [1, 2]]
            delta_notifications = [pd.Timedelta(minutes=60)]
            betas = [0.5, 0.7]
            gamma_buffers = [pd.Timedelta(minutes=60)]
        elif scenario == "DAM":
            latencies = [60 * 15, 60 * 30]
            flex_windows = [pd.Timedelta(hours=h) for h in [4, 6]]
            delta_notifications = [pd.Timedelta(minutes=60 * 12), pd.Timedelta(minutes=60 * 24)]
            betas = [0.7]
            gamma_buffers = [pd.Timedelta(minutes=120)]
        else:
            raise ValueError(f"SENSITIVE mode not defined for scenario: {scenario}")
    else:
        raise ValueError(f"Unknown simulation mode: {simulation}")

    return {
        'latencies': latencies,
        'flex_windows': flex_windows,
        'delta_notifications': delta_notifications,
        'betas': betas,
        'gamma_buffers': gamma_buffers,
    }


# ---------------------------------------------------------------------------
# Full sweep wrapper: run LAD-FLEX over an hourly grid of flexibility windows
# ---------------------------------------------------------------------------
def run_lad_flex_sweep(df_tasks, df_power_energy, dates, params):
    """
    Run LAD-FLEX over every (parameter combination x hourly date) and assemble the
    summary table and the detailed results dictionary.

    Parameters
    ----------
    df_tasks : pd.DataFrame
        Task table (needs 'start_time', 'end_time', 'wait_time', 'runtime_i',
        'plan_power').
    df_power_energy : pd.DataFrame
        Total DC power/energy series for `df_tasks` (output of
        `fast_power_energy_series`). Used for the total-energy and rebound KPIs.
    dates : iterable of pd.Timestamp
        Flexibility-window start times t1 (typically an hourly date_range).
    params : dict
        As returned by `return_params` (latencies, flex_windows,
        delta_notifications, betas, gamma_buffers).

    Returns
    -------
    (df_summary, res_dfs)
        df_summary : multi-indexed KPI table
            ['timestamp','latency','flex_window','delta_notification','beta','gamma_buffer']
        res_dfs : dict keyed by
            (date, latency, flex_window, delta_notification, beta, gamma_buffer)
    """
    latencies = params['latencies']
    flex_windows = params['flex_windows']
    delta_notifications = params['delta_notifications']
    betas = params['betas']
    gamma_buffers = params['gamma_buffers']

    res_dfs = {}
    summary_data = []

    for latency in latencies:
        for flex_window in flex_windows:
            for beta in betas:
                for gamma_buffer in gamma_buffers:
                    for delta_notification in delta_notifications:
                        for date in dates:
                            t1 = date
                            t2 = date + flex_window
                            t0 = t1 - delta_notification
                            tend = t2 + gamma_buffer

                            key = (date, latency, flex_window, delta_notification, beta, gamma_buffer)

                            df_deferrable, df_not_deferrable = LAD_flex(
                                df_tasks, latency, t1, t2, beta, delta_notification, gamma_buffer)

                            timeline_deferrable = get_timeline(df_deferrable)
                            if len(timeline_deferrable) >= 2:
                                df_deferrable_power_energy = fast_power_energy_series(
                                    df_deferrable, timeline_deferrable)
                            else:
                                # No deferrable tasks -> empty power/energy series
                                df_deferrable_power_energy = pd.DataFrame(
                                    columns=['power_kW', 'energy_kWh'],
                                    index=pd.DatetimeIndex([], name='time'))

                            energy_window = df_deferrable_power_energy[
                                (df_deferrable_power_energy.index >= t1) &
                                (df_deferrable_power_energy.index < t2)]
                            df_total_power_energy = df_power_energy[
                                (df_power_energy.index >= t1) & (df_power_energy.index < t2)]

                            df_t0_to_tend_def = df_deferrable_power_energy[
                                (df_deferrable_power_energy.index >= t0) &
                                (df_deferrable_power_energy.index < tend)]
                            df_t0_to_tend_total = df_power_energy[
                                (df_power_energy.index >= t0) & (df_power_energy.index < tend)]

                            rebound_mask = (
                                ((df_power_energy.index >= t0) & (df_power_energy.index < t1)) |
                                ((df_power_energy.index >= t2) & (df_power_energy.index < tend)))
                            rebound_df = df_power_energy[rebound_mask]

                            deferrable_energy_sum = energy_window['energy_kWh'].sum()
                            total_energy_sum = df_total_power_energy['energy_kWh'].sum()
                            percentage_flexible = (
                                deferrable_energy_sum / total_energy_sum * 100
                                if total_energy_sum != 0 else 0)
                            rebound_energy_sum = rebound_df['energy_kWh'].sum()

                            res_dfs[key] = {
                                't1': t1, 't2': t2, 't0': t0, 'tend': tend,
                                'latency': latency, 'beta': beta,
                                'delta_notification': delta_notification,
                                'gamma_buffer': gamma_buffer,
                                'df_deferrable': df_deferrable,
                                'df_not_deferrable': df_not_deferrable,
                                'df_deferrable_power_energy': df_deferrable_power_energy,
                                'deferrable_energy_window': energy_window,
                                'df_total_power_energy': df_total_power_energy,
                                'df_t0_to_tend_def': df_t0_to_tend_def,
                                'df_t0_to_tend_total': df_t0_to_tend_total,
                                'df_rebound_energy': rebound_df,
                            }

                            summary_data.append({
                                'timestamp': date,
                                'latency': latency,
                                'flex_window': flex_window,
                                'delta_notification': delta_notification,
                                'beta': beta,
                                'gamma_buffer': gamma_buffer,
                                'deferrable_energy_kWh': deferrable_energy_sum,
                                'total_energy_kWh': total_energy_sum,
                                'percentage_flexible_energy': percentage_flexible,
                                'deferrable_task_count': len(df_deferrable),
                                'rebound_energy_kWh': rebound_energy_sum,
                            })

    df_summary = pd.DataFrame(summary_data)
    df_summary.set_index(
        ['timestamp', 'latency', 'flex_window', 'delta_notification', 'beta', 'gamma_buffer'],
        inplace=True)
    return df_summary, res_dfs


# ---------------------------------------------------------------------------
# Notable points (top-3 / 75th-pct / median deferrable-energy windows)
# ---------------------------------------------------------------------------
def deferrable_energy_timeseries(df_summary):
    """
    For a DEFAULT (single value per parameter) sweep, return deferrable_energy_kWh
    as a Series indexed by timestamp and sorted in time.
    """
    s = df_summary['deferrable_energy_kWh'].copy()
    s.index = df_summary.index.get_level_values('timestamp')
    return s.sort_index()


def compute_notable_points(df_summary):
    """
    Identify nine notable windows from a DEFAULT sweep's deferrable-energy series:
      P1-P3 : the three highest-energy windows
      Q1-Q3 : the three windows closest to the 75th percentile
      R1-R3 : the three windows closest to the median (50th percentile)

    Returns a dict {'p1':ts, ..., 'r3':ts} of timestamps.
    """
    s = deferrable_energy_timeseries(df_summary)
    s_sorted = s.sort_values(ascending=False)

    p1, p2, p3 = s_sorted.index[:3]
    q75 = s_sorted.quantile(0.75)
    q1, q2, q3 = (s_sorted - q75).abs().sort_values().index[:3]
    q50 = s_sorted.quantile(0.50)
    r1, r2, r3 = (s_sorted - q50).abs().sort_values().index[:3]

    return {'p1': p1, 'p2': p2, 'p3': p3,
            'q1': q1, 'q2': q2, 'q3': q3,
            'r1': r1, 'r2': r2, 'r3': r3}


# ---------------------------------------------------------------------------
# Deferability confusion matrix (uncertainty analysis)
# ---------------------------------------------------------------------------
def extract_task_decisions(res_dfs, start_date=None, end_date=None):
    """
    Collect the set of deferrable / not-deferrable task identities across all
    windows in a res_dfs dict. A task identity is (task_name, start_time), where
    task_name is the DataFrame index of df_deferrable / df_not_deferrable.

    If start_date / end_date are given, only windows whose t1 (key[0]) falls in
    [start_date, end_date) are considered.

    Returns dict with 'deferrable', 'not_deferrable', 'all_tasks' (sets).
    """
    deferrable, not_deferrable = set(), set()
    for key, value in res_dfs.items():
        if start_date is not None and not (start_date <= key[0] < end_date):
            continue
        for name, row in value['df_deferrable'].iterrows():
            deferrable.add((name, row['start_time']))
        for name, row in value['df_not_deferrable'].iterrows():
            not_deferrable.add((name, row['start_time']))
    return {'deferrable': deferrable,
            'not_deferrable': not_deferrable,
            'all_tasks': deferrable | not_deferrable}


def build_confusion_matrix(baseline_tasks, comparison_tasks):
    """
    Confusion matrix of deferability decisions: baseline (zeta=0) as "actual",
    comparison (zeta>0) as "predicted".
      TP = both deferrable, FP = only comparison deferrable,
      FN = only baseline deferrable, TN = neither deferrable.
    Returns dict with TP/FP/FN/TN counts.
    """
    all_tasks = baseline_tasks['all_tasks'] | comparison_tasks['all_tasks']
    b_def = baseline_tasks['deferrable']
    c_def = comparison_tasks['deferrable']
    TP = b_def & c_def
    FP = c_def - b_def
    FN = b_def - c_def
    TN = (all_tasks - b_def) & (all_tasks - c_def)
    return {'TP': len(TP), 'FP': len(FP), 'FN': len(FN), 'TN': len(TN)}


def calculate_metrics(cm):
    """Accuracy / precision / recall / F1 / specificity from a confusion-matrix dict."""
    TP, FP, FN, TN = cm['TP'], cm['FP'], cm['FN'], cm['TN']
    total = TP + FP + FN + TN
    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    specificity = TN / (TN + FP) if (TN + FP) else 0
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1_score': f1, 'specificity': specificity}


# ---------------------------------------------------------------------------
# Before / after plotting helper
# ---------------------------------------------------------------------------
def plot_ladflex_before_after(point_name, points_dates, res_dfs, latencies,
                              fixed_flex_window, fixed_delta_notification,
                              fixed_beta, fixed_gamma_buffer,
                              save_path=None, show=True, title_string=None):
    """
    LAD-Flex before/after plot: DC total power vs power after deferral, with the
    notification / flexible / rebound regions shaded.
    """
    tuple_point = (points_dates[point_name], latencies[0], fixed_flex_window,
                   fixed_delta_notification, fixed_beta, fixed_gamma_buffer)

    t_0 = res_dfs[tuple_point]['t0']
    t_1 = res_dfs[tuple_point]['t1']
    t_2 = res_dfs[tuple_point]['t2']
    t_end = res_dfs[tuple_point]['tend']

    power_def = res_dfs[tuple_point]['df_t0_to_tend_def'].power_kW
    power_total = res_dfs[tuple_point]['df_t0_to_tend_total'].power_kW

    common_index = power_def.index.union(power_total.index)
    power_def_aligned = power_def.reindex(common_index, method='nearest')
    power_total_aligned = power_total.reindex(common_index, method='nearest')
    power_diff = power_total_aligned - power_def_aligned

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.step(power_total.index, power_total.values, where='post',
            label='DC Total Power', color='blue', linewidth=2)
    ax.step(power_diff.index, power_diff.values, where='post', linestyle='--',
            color='purple', label='DC Power after LAD-Flex', linewidth=2)

    ax.fill_between(power_total.index, power_total.values, power_diff.values,
                    where=(power_total.index <= t_1), color='lightblue', alpha=0.5)
    ax.fill_between(power_total.index, power_total.values, power_diff.values,
                    where=((power_total.index > t_1) & (power_total.index <= t_2)),
                    color='lightgreen', alpha=0.5, label='Flexible energy')
    ax.fill_between(power_total.index, power_total.values, power_diff.values,
                    where=(power_total.index > t_2), color='lightblue', alpha=0.5,
                    label='Rebound energy')

    for t, label in zip([t_0, t_1, t_2, t_end], ['$t_0$', '$t_1$', '$t_2$', '$t_{end}$']):
        ax.axvline(t, linestyle='--', color='red', alpha=0.3, linewidth=2)
        ax.text(t, -0.07, label, color="red", ha='center', va='top', fontsize=16,
                transform=ax.get_xaxis_transform(), clip_on=False)

    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3)
    if title_string:
        ax.set_title(f'{title_string}: LAD-Flex Before and After', fontsize=18)
    else:
        ax.set_title(f'{point_name.upper()}: LAD-Flex Before and After', fontsize=18)
    ax.set_xlabel('Time', fontsize=16)
    ax.set_ylabel('Power (kW)', fontsize=16)
    ax.legend(fontsize=14, loc="best")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    return fig, ax
