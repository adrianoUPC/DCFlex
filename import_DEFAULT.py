# %% Imports and Configuration
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import matplotlib.dates as mdates
import seaborn as sns
from utils import *
from get_parameters import *

SIMULATION = "DEFAULT"
SCENARIO = "DAM"

# %% Load Data
# Load summary DataFrame
filename = "df_summary_" + SIMULATION + "_" + SCENARIO + ".joblib"
df_summary = joblib.load(filename)
"""
df_summary: pd.DataFrame
    Summary statistics for LAD-FLEX simulations across all parameter combinations.

    Structure:
        MultiIndex with levels:
            - timestamp: pd.Timestamp - Date/time of the simulation window
            - latency: int - Task latency in seconds
            - flex_window: pd.Timedelta - Duration of flexibility window
            - delta_notification: pd.Timedelta - Notification time before flexibility window
            - beta: float - Sensitivity parameter for deferral decision
            - gamma_buffer: pd.Timedelta - Buffer time after flexibility window

        Columns:
            - deferrable_energy_kWh: float - Energy from deferrable tasks in the flex window
            - total_energy_kWh: float - Total energy consumption in the flex window
            - percentage_flexible_energy: float - Percentage of flexible energy (0-100)
            - deferrable_task_count: int - Number of tasks that were deferred
            - rebound_energy_kWh: float - Energy consumption in rebound periods (t0-t1, t2-tend)
"""

# Load detailed results dictionary
filename = "res_dfs_" + SIMULATION + "_" + SCENARIO + ".joblib"
res_dfs = joblib.load(filename)
"""
res_dfs: dict
    Detailed simulation results for each parameter combination.

    Structure:
        Dictionary with keys: (date, latency, flex_window, delta_notification, beta, gamma_buffer)

        Each value is a dictionary containing:

            Metadata:
                - t1: pd.Timestamp - Start of flexibility window
                - t2: pd.Timestamp - End of flexibility window
                - t0: pd.Timestamp - Start of notification period (t1 - delta_notification)
                - tend: pd.Timestamp - End of buffer period (t2 + gamma_buffer)
                - latency: int - Task latency in seconds
                - beta: float - Sensitivity parameter
                - delta_notification: pd.Timedelta - Notification time
                - gamma_buffer: pd.Timedelta - Buffer time

            DataFrames:
                - df_deferrable: pd.DataFrame - Tasks identified as deferrable
                - df_not_deferrable: pd.DataFrame - Tasks that are not deferrable
                - df_deferrable_power_energy: pd.DataFrame - Power/energy time series for deferrable tasks
                - deferrable_energy_window: pd.DataFrame - Deferrable energy within [t1, t2)
                - df_total_power_energy: pd.DataFrame - Total power/energy within [t1, t2)
                - df_t0_to_tend_def: pd.DataFrame - Deferrable power/energy for extended period [t0, tend)
                - df_t0_to_tend_total: pd.DataFrame - Total power/energy for extended period [t0, tend)
                - df_rebound_energy: pd.DataFrame - Energy in rebound periods [t0, t1) and [t2, tend)
"""

# %% Extract Parameters
params = return_params(SIMULATION, SCENARIO)
latencies = params['latencies']
flex_windows = params['flex_windows']
delta_notifications = params['delta_notifications']
betas = params['betas']
gamma_buffers = params['gamma_buffers']

# %% Prepare Sample Week Data
sample_week = pd.date_range(start='1970-02-01', periods=7, freq='D')
start_date = sample_week[0]
end_date = sample_week[-1] + pd.Timedelta(days=1)  # Include the last day

df_summary_week = df_summary.loc[start_date:end_date]

# %% PLOT 1: deferrable energy plot for every latency
fixed_flex_window = df_summary.index.get_level_values('flex_window').unique()[0]
fixed_delta_notification = df_summary.index.get_level_values('delta_notification').unique()[0]
fixed_beta = df_summary.index.get_level_values('beta').unique()[0]
fixed_gamma_buffer = df_summary.index.get_level_values('gamma_buffer').unique()[0]

fig, ax = plt.subplots(figsize=(14, 6))

for latency in df_summary_week.index.get_level_values('latency').unique():
    # Extract the subset for this parameter combination. (for default only one latency)
    df_plot = df_summary_week.xs(
        (latency, fixed_flex_window, fixed_delta_notification, fixed_beta, fixed_gamma_buffer),
        level=('latency', 'flex_window', 'delta_notification', 'beta', 'gamma_buffer')
    ).sort_index()

    if df_plot.empty:
        continue

    ax.plot(
        df_plot.index,
        df_plot['deferrable_energy_kWh'],
        marker='o',
        label=f'latency={latency}'
    )

# Finalize the plot
ax.set_xlabel('Time')
ax.set_ylabel('Deferrable Energy (kWh)')
ax.set_title(
    f'LAD-Flex for the chosen sample week\nflex_window={fixed_flex_window}, delta_notification={fixed_delta_notification}')
# ax.legend(title='LATENCY')
ax.grid(True)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#keep flex window fixed and 