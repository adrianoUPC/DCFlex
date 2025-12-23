import pandas as pd
import numpy as np


def get_timeline(df):
    """
    Get the timeline of start and end times from the DataFrame.
    """
    return pd.concat([df['start_time'], df['end_time']]).sort_values().unique()

def build_task_intervals(df):
    return pd.IntervalIndex.from_arrays(df['start_time'], df['end_time'], closed='left')

def fast_power_energy_series(df, timeline):
    """
    Efficiently compute power and energy series over the timeline using IntervalIndex.
    Returns a DataFrame with 'time', 'power_kW', and 'energy_kWh'.
    """
    intervals = pd.IntervalIndex.from_arrays(timeline[:-1], timeline[1:], closed='left')
    task_intervals = build_task_intervals(df)
    power_series = []
    energy_series = []

    for interval in intervals:
        overlapping = task_intervals.overlaps(interval)
        active_power = df.loc[overlapping, 'plan_power'].sum() / 1000  # kW
        duration_h = (interval.right - interval.left).total_seconds() / 3600
        energy_kWh = active_power * duration_h
        power_series.append(active_power)
        energy_series.append(energy_kWh)

    df_res = pd.DataFrame({
        'time': timeline[:-1],
        'power_kW': power_series,
        'energy_kWh': energy_series
    })

    #set the time as datetime
    df_res['time'] = pd.to_datetime(df_res['time'])
    df_res.set_index('time', inplace=True)
    return df_res

def LAD_flex(df_a, LATENCY, t1, t2, beta_overhang, delta_notification, gamma_buffer):
    """
    Perform LAD Flex analysis on the DataFrame df_a to get the deferrable tasks.
    :param df_a: dataframe containing task information
    :param LATENCY: minimum latency for deferrable tasks
    :param t1: start of flex window
    :param t2: end of flex window
    :param beta_overhang: parameter for overhang calculation
    :param delta_notification: parameter to define the notification period
    :param gamma_buffer: parameter for final buffer calculation
    :return: df_a_flex: DataFrame with deferrable tasks
    """

    # Filter tasks based on latency threshold
    df_flex = df_a[df_a['wait_time'] >= LATENCY].copy()

    # Define whiskers around t1 and t2
    t0 = t1 - delta_notification
    tend = t2 + gamma_buffer

    # Select only tasks that lie entirely or partially within the whiskered window
    # This reduces the number of tasks we have to examine closely
    task_intervals = pd.IntervalIndex.from_arrays(df_flex['start_time'], df_flex['end_time'], closed='left')
    window_interval = pd.Interval(left=t0, right=tend, closed='left')
    overlaps_mask = task_intervals.overlaps(window_interval)
    df_flex = df_flex[overlaps_mask].copy()

    # Apply deferrability condition with beta-overhang allowance
    df_flex["Deferrable"] = (
        (df_flex['start_time'] >= t0) &
        (df_flex['end_time'] <= tend) &
        ((t1 - df_flex['start_time']).dt.total_seconds() <= beta_overhang * df_flex['runtime_i']) & #if beta = 0.5, half of the task can be "rebound" effect
        ((df_flex['end_time'] - t2).dt.total_seconds() <= beta_overhang * df_flex['runtime_i'])
    )

    # Split deferrable and non-deferrable
    df_deferrable = df_flex[df_flex["Deferrable"]].copy()
    df_not_deferrable = df_flex[~df_flex["Deferrable"]].copy()

    return df_deferrable, df_not_deferrable

def Value(delta):
    """
    Parameters
    ----------
    delta : pd.Timedelta
        Notification period duration (time between t0 and t1)

    Returns
    -------
    float
        Flexibility value in €/kWh

    Pricing Tiers
    -------------
    - δ ≤ 5 min:   0.25 €/kWh  (Real-time reserves - highest value)
    - 5 < δ ≤ 30:  0.12 €/kWh  (Fast frequency reserves - aFRR/mFRR)
    - 30 < δ ≤ 60: 0.06 €/kWh  (Slow frequency reserves - sRR)
    - δ > 60:      0.03 €/kWh  (Energy shifting - lowest value)

    Rationale
    ---------
    Shorter notification periods provide more valuable grid services:
    - Real-time reserves (≤5 min): Respond immediately to frequency deviations
    - Fast reserves (≤30 min): Balance supply/demand on short timescales
    - Slow reserves (≤60 min): Longer-term load balancing
    - Energy shifting (>60 min): Optimize around price/carbon signals

    Usage in Optimization
    ---------------------
    Used to compute revenue: R(δ) = F(δ) × V(δ)
    where F(δ) is flexibility amount (kWh) at notification time δ.
    The optimal notification period δ* maximizes R(δ).
    """
    delta = delta.total_seconds() / 60  # Convert to minutes
    if delta <= 5:
        return 0.25  # €/kWh - realistic high value
    elif delta <= 30:
        return 0.12
    elif delta <= 60:
        return 0.06
    else:
        return 0.03
    
    
def apply_duration_uncertainty(df, gamma, random_seed=42):
    """
    Apply normally distributed prediction error to task durations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe with 'runtime_i' and 'start_time'
    gamma : float
        Relative prediction error (e.g., 0.2 for 20% error)
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Copy of dataframe with perturbed 'end_time' and new 'duration_predicted' column
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Create a copy to avoid modifying original
    df_perturbed = df.copy()
    
    # Actual duration in seconds
    d_actual = df_perturbed["runtime_i"].values
    
    # Generate prediction errors: ε_k ~ N(0, (γ * d_k)²)
    sigma = gamma * d_actual
    epsilon = np.random.normal(loc=0, scale=sigma, size=len(d_actual))
    
    # Predicted duration: d̂_k = d_k + ε_k
    d_predicted = d_actual + epsilon
    
    # Ensure predicted durations are positive (clip at 1 second minimum)
    d_predicted = np.maximum(d_predicted, 1.0)
    
    # Store predicted duration
    df_perturbed["duration_predicted"] = d_predicted
    df_perturbed["duration_actual"] = d_actual
    df_perturbed["prediction_error"] = epsilon
    df_perturbed["prediction_error_pct"] = (epsilon / d_actual) * 100
    
    # Modify end_time based on predicted duration
    df_perturbed["end_time_actual"] = df_perturbed["end_time"]  # Keep original
    df_perturbed["end_time"] = df_perturbed["start_time"] + pd.to_timedelta(d_predicted, unit="s") + pd.to_timedelta(df_perturbed["wait_time"], unit="s")
    
    return df_perturbed