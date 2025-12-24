# %% Imports and Configuration
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import matplotlib.dates as mdates
import seaborn as sns
from utils import *
from get_parameters import *
from utils_plotting import *

SIMULATION = "DEFAULT"
SCENARIO = "BASELINE" # choose from: BASELINE, mFRR, DAM
ZETA = 0  # Task duration uncertainty level: 0, 0.10, 0.20, 0.30, 0.50, or 0.70

# Format ZETA suffix for filenames
if ZETA == 0:
    zeta_suffix = "ZETA_0"
elif ZETA == 0.10:
    zeta_suffix = "ZETA_010"
elif ZETA == 0.20:
    zeta_suffix = "ZETA_020"
elif ZETA == 0.30:
    zeta_suffix = "ZETA_030"
elif ZETA == 0.50:
    zeta_suffix = "ZETA_050"
elif ZETA == 0.70:
    zeta_suffix = "ZETA_070"

# Create plot subfolder for this scenario configuration
import os
plot_subfolder = f"PLOTS/{SIMULATION}_{SCENARIO}_{zeta_suffix}"
os.makedirs(plot_subfolder, exist_ok=True)

# %% Load Data
# Load summary DataFrame
filename = f"EXPORTS/df_summary_{SIMULATION}_{SCENARIO}_{zeta_suffix}.joblib"
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
filename = f"EXPORTS/res_dfs_{SIMULATION}_{SCENARIO}_{zeta_suffix}.joblib"
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
latency = latencies[0]

df_filtered = df_summary_week.xs(
    (fixed_flex_window, fixed_delta_notification, fixed_beta, fixed_gamma_buffer, latency),
    level=('flex_window', 'delta_notification', 'beta', 'gamma_buffer', 'latency')
)

df_sorted = df_filtered.sort_values(by='deferrable_energy_kWh', ascending=False)
#OBS: this df_sorted is still a multi-index dataframe


top_3 = df_sorted.head(3)
p1, p2, p3 = top_3.index[:3]
p1_val, p2_val, p3_val = top_3['deferrable_energy_kWh'].iloc[:3]

# --- Compute quantiles ---
q75_val = df_sorted['deferrable_energy_kWh'].quantile(0.75)
q50_val = df_sorted['deferrable_energy_kWh'].quantile(0.50)

# --- Find 3 closest values to 75th percentile ---
closest_to_75 = (df_sorted['deferrable_energy_kWh'] - q75_val).abs().sort_values().head(3)
q1, q2, q3 = closest_to_75.index[:3]
q1_val, q2_val, q3_val = df_sorted.loc[[q1, q2, q3], 'deferrable_energy_kWh'].values

# --- Find 3 closest values to 50th percentile ---
closest_to_50 = (df_sorted['deferrable_energy_kWh'] - q50_val).abs().sort_values().head(3)
r1, r2, r3 = closest_to_50.index[:3]
r1_val, r2_val, r3_val = df_sorted.loc[[r1, r2, r3], 'deferrable_energy_kWh'].values


# %%PLOT 2: Deferrable energy for the notable points LARGE
fig, ax = plt.subplots(figsize=(14, 6))

# Plot notable points
ax.scatter([p1, p2, p3], [p1_val, p2_val, p3_val], color='red', marker='^', s=150, zorder=5,
           label='$P_1$, $P_2$, $P_3$')
ax.scatter([q1, q2, q3], [q1_val, q2_val, q3_val], color='orange', marker='^', s=150, zorder=5,
           label='$Q_1$, $Q_2$, $Q_3$')
ax.scatter([r1, r2, r3], [r1_val, r2_val, r3_val], color='green', marker='^', s=150, zorder=5,
           label='$R_1$, $R_2$, $R_3$')

# Plot the deferrable energy curve
df_plot = df_summary_week.xs(
    (latencies[0], fixed_flex_window, fixed_delta_notification, fixed_beta, fixed_gamma_buffer),
    level=('latency', 'flex_window', 'delta_notification', 'beta', 'gamma_buffer')
).sort_index()

ax.plot(
    df_plot.index,
    df_plot['deferrable_energy_kWh'],
    marker='o',
    linewidth=2,
)

# Axis labels and title with larger font
ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Deferrable Energy (kWh)', fontsize=16)

# Format window and notification as string
def format_timedelta_h(td):
    return f"{int(td.total_seconds() // 3600)} h"
flex_str = format_timedelta_h(fixed_flex_window)
notif_str = format_timedelta_h(fixed_delta_notification)

# Set title
ax.set_title(
    f"Baseline Scenario: LAD-Flex for the Chosen Sample Week\n"
    f"$\hat \lambda$ = {latency}, $\beta$ = {fixed_beta} $\eta \Delta t$ = {flex_str}, $\delta$ = {notif_str}",
    fontsize=18
)

ax.tick_params(axis='both', labelsize=14)
ax.legend(fontsize=14)
ax.grid(True)
plt.xticks(rotation=0)
plt.tight_layout()

# Save and show
plt.savefig(f'{plot_subfolder}/deferrable_energy_notable_points_LARGE_{SIMULATION}_{SCENARIO}_{zeta_suffix}.png', dpi=300)
plt.show()

# %%Extracting data to get 3x3 PLOTS
points_dates = {
    "p1": p1, "p2": p2, "p3": p3,
    "q1": q1, "q2": q2, "q3": q3,
    "r1": r1, "r2": r2, "r3": r3
}

joblib.dump(points_dates, f"EXPORTS/points_dates_{SIMULATION}_{SCENARIO}_{zeta_suffix}.joblib")



# %%3x3 plot - LARGE FONTS VERSION
# Use seaborn style
sns.set_theme(style="whitegrid")

fig, axs = plt.subplots(3, 3, figsize=(20, 14), sharex=False, sharey=False)

# Define point ordering for 3x3 grid
point_names = ['p1', 'p2', 'p3', 'q1', 'q2', 'q3', 'r1', 'r2', 'r3']

# Collect handles for unified legend
all_handles = []
all_labels = []

# Color definitions (colorblind-friendly)
notif_color = '#A6D854'  # soft green
flex_color = '#FC9272'   # soft red
line_color = '#0571B0'   # blue

# Plot each point
for idx, point_name in enumerate(point_names):
    row, col = divmod(idx, 3)
    ax = axs[row, col]

    # Get data for this point
    tuple_point = (points_dates[point_name], latencies[0], fixed_flex_window,
                   fixed_delta_notification, fixed_beta, fixed_gamma_buffer)

    t0 = res_dfs[tuple_point]['t0']
    t1 = res_dfs[tuple_point]['t1']
    t2 = res_dfs[tuple_point]['t2']
    tend = res_dfs[tuple_point]['tend']

    power_t0_tend_def = res_dfs[tuple_point]['df_t0_to_tend_def'].power_kW

    # Background shading
    h1 = ax.axvspan(t0, t1, color=notif_color, alpha=0.3, label='Notification Period')
    h2 = ax.axvspan(t1, t2, color=flex_color, alpha=0.3, label='Flexibility Window')

    # Dashed lines
    ax.axvline(t0, linestyle='--', color=notif_color, alpha=0.6)
    ax.axvline(t1, linestyle='--', color=flex_color, alpha=0.6)
    ax.axvline(t2, linestyle='--', color=flex_color, alpha=0.6)

    # Power curve
    h3, = ax.step(power_t0_tend_def.index, power_t0_tend_def.values, where='post',
                  color=line_color, linewidth=2.2, label='Deferred Power')

    # Formatting
    ax.set_title(f"{point_name.upper()}: " + str(points_dates[point_name].date()),
                 fontweight='bold', fontsize=19)
    ax.set_xlabel('Time', fontsize=17)
    ax.set_ylabel('Power (kW)', fontsize=17)
    ax.tick_params(axis='x', rotation=0, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Date formatting on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Show every hour

    # Thicker, visible spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.4)

    # Collect legend handles from the first plot only
    if idx == 0:
        all_handles.extend([h1, h2, h3])
        all_labels.extend(['Notification Period', 'Flexibility Window', 'Deferred Power'])

# Shared legend at the bottom
fig.legend(all_handles, all_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.015), fontsize=18)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.16)

# Save + show
plt.savefig(f'{plot_subfolder}/power_consumption_3x3_LARGE_{SIMULATION}_{SCENARIO}.png', dpi=300)
plt.savefig(f'{plot_subfolder}/power_consumption_3x3_LARGE_{SIMULATION}_{SCENARIO}.pdf', dpi=300)
plt.show()


# %%TOTAL DC VS AFTER FLEX ACTIVATION
# Plot P1 using the utility function (from utils_plotting.py)
plot_ladflex_before_after(
    point_name='p1',
    points_dates=points_dates,
    res_dfs=res_dfs,
    latencies=latencies,
    fixed_flex_window=fixed_flex_window,
    fixed_delta_notification=fixed_delta_notification,
    fixed_beta=fixed_beta,
    fixed_gamma_buffer=fixed_gamma_buffer,
    save_path=f'{plot_subfolder}/LAD_flex_P1_explanation_{SIMULATION}_{SCENARIO}.png',
    show=True
)

# %%KDE plot
energy_values = []

for latency in df_summary_week.index.get_level_values('latency').unique():
    df_plot = df_summary_week.xs(
        (latency, fixed_flex_window, fixed_delta_notification, fixed_beta, fixed_gamma_buffer),
        level=('latency', 'flex_window', 'delta_notification', 'beta', 'gamma_buffer')
    ).sort_index()

    if df_plot.empty:
        continue

    energy_values.extend(df_plot['deferrable_energy_kWh'].values)

# Convert to pandas Series for convenience
energy_series = pd.Series(energy_values)

# Plot KDE
plt.figure(figsize=(10, 5))
sns.kdeplot(energy_series, fill=True, bw_adjust=0.3, clip=(0, None))
plt.title('KDE of Deferrable Energy (kWh)')
plt.xlabel('Deferrable Energy (kWh)')
plt.ylabel('Density')
plt.grid(True)
plt.tight_layout()
# plt.savefig(f'{plot_subfolder}/KDE_deferrable_energy.png', dpi=300)
plt.show()

# %% Table: Breakdown of task and energy flexibility by point

# Create table data
table_data = []

for point_name in ['p1', 'p2', 'p3', 'q1', 'q2', 'q3', 'r1', 'r2', 'r3']:
    point_date = points_dates[point_name]
    
    # Extract data from df_summary for this point
    point_data = df_summary.loc[
        (point_date, latencies[0], fixed_flex_window, 
         fixed_delta_notification, fixed_beta, fixed_gamma_buffer)
    ]
    
    table_data.append({
        'Point': point_name.upper(),
        'Tasks Count': int(point_data['deferrable_task_count']),
        'Deferrable (kWh)': point_data['deferrable_energy_kWh'],
        'Total (kWh)': point_data['total_energy_kWh'],
        'Percentage Def. (%)': point_data['percentage_flexible_energy']
    })

# Create DataFrame
df_table = pd.DataFrame(table_data)

# Display table
print("\nTable: Breakdown of task and energy flexibility by point")
print("="*70)
print(df_table.to_string(index=False))
print("="*70)

# Save to CSV
df_table.to_csv(f'{plot_subfolder}/task_energy_breakdown_{SIMULATION}_{SCENARIO}_{zeta_suffix}.csv', index=False)

