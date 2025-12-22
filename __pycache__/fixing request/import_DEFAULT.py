# %% Imports and Configuration
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import matplotlib.dates as mdates
import seaborn as sns
from utils import *
from get_parameters import *

SIMULATION = "DEFAULT"
SCENARIO = "BASELINE"

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

# %% Selecting the notable points for the analysis

df_filtered = df_summary_week.xs(
    (fixed_flex_window, fixed_delta_notification, fixed_beta, fixed_gamma_buffer, latencies[0]),
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

# %%# %%PLOT 2: Defferable energy for the notable points
fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter([p1, p2, p3], [p1_val, p2_val, p3_val], color='red', marker='^', s=150, zorder=5,
           label='$P_1$, $P_2$, $P_3$')
ax.scatter([q1, q2, q3], [q1_val, q2_val, q3_val], color='blue', marker='^', s=150, zorder=5,
           label='$Q_1$, $Q_2$, $Q_3$')
ax.scatter([r1, r2, r3], [r1_val, r2_val, r3_val], color='green', marker='^', s=150, zorder=5,
           label='$R_1$, $R_2$, $R_3$')

df_plot = df_summary_week.xs(
    (latency, fixed_flex_window, fixed_delta_notification, fixed_beta, fixed_gamma_buffer),
    level=('latency', 'flex_window', 'delta_notification', 'beta', 'gamma_buffer')
).sort_index()

ax.plot(
    df_plot.index,
    df_plot['deferrable_energy_kWh'],
    marker='o',
    # label=f'LATENCY={latency}'
)

ax.set_xlabel('Time')
ax.set_ylabel('Deferrable Energy (kWh)')

def format_timedelta_h(td):
    return f"{int(td.total_seconds() // 3600)} h"
flex_str = format_timedelta_h(fixed_flex_window)
notif_str = format_timedelta_h(fixed_delta_notification)
ax.set_title(
    f"LAD-Flex for the chosen sample week\n"
    f"$\eta \Delta t$ = {flex_str}, $\delta$ = {notif_str}"
)

ax.legend()
ax.grid(True)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


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
    (latency, fixed_flex_window, fixed_delta_notification, fixed_beta, fixed_gamma_buffer),
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
    f"LAD-Flex for the Chosen Sample Week\n"
    f"$\eta \Delta t$ = {flex_str}, $\delta$ = {notif_str}",
    fontsize=18
)

# Tick label size
ax.tick_params(axis='both', labelsize=14)

# Legend with larger font
ax.legend(fontsize=14)

# Grid and layout
ax.grid(True)
plt.xticks(rotation=0)
plt.tight_layout()

# Save and show
plt.show()

# %%Extracting data to get 3x3 PLOTS
points_dates = {
    "p1": p1, "p2": p2, "p3": p3,
    "q1": q1, "q2": q2, "q3": q3,
    "r1": r1, "r2": r2, "r3": r3
}

joblib.dump(points_dates, "points_dates_" + SIMULATION + "_" + SCENARIO + ".joblib")


# %% Plot for P1 only
point = p1
tuple_p1 = (point, latencies[0], fixed_flex_window, fixed_delta_notification, fixed_beta, fixed_gamma_buffer)

t0 = res_dfs[tuple_p1]['t0']
t1 = res_dfs[tuple_p1]['t1']
t2 = res_dfs[tuple_p1]['t2']
tend = res_dfs[tuple_p1]['tend']

power_t0_tend_def = res_dfs[tuple_p1]['df_t0_to_tend_def'].power_kW
power_t0_tend_total = res_dfs[tuple_p1]['df_t0_to_tend_total'].power_kW

# %%
fig, ax = plt.subplots(figsize=(14, 6))
ax.axvspan(t0, t1, color='green', alpha=0.3, label='Notification Period')
plt.step(power_t0_tend_def.index, power_t0_tend_def.values, label='Deferred Tasks Power Consumption', where='post')
ax.axvspan(t1, t2, color='red', alpha=0.3, label='Flexibility Window')
# plt.step(power_t0_tend_total.index, power_t0_tend_total.values, label='Total Power Consumption', where='post')
ax.set_title('Deferred Tasks Power Consumption Over Time')
ax.set_xlabel('Time')
ax.set_ylabel('Total Power (kW)')
ax.legend(fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig("LAD_flex_default_plot.png", dpi=300)
plt.show()

# %%3x3 plot - Publication Quality
# Set publication-quality parameters
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
})

# Create 3x3 subplot grid
fig, axs = plt.subplots(3, 3, figsize=(18, 12), sharex=False, sharey=False)

# Define point ordering for 3x3 grid
point_names = ['p1', 'p2', 'p3', 'q1', 'q2', 'q3', 'r1', 'r2', 'r3']

all_handles = []
all_labels = []

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
    # power_t0_tend_total = res_dfs[tuple_point]['df_t0_to_tend_total'].power_kW

    # Plot notification period and flexibility window
    h1 = ax.axvspan(t0, t1, color='green', alpha=0.3, label='Notification Period')
    h2 = ax.axvspan(t1, t2, color='red', alpha=0.3, label='Flexibility Window')
    h3, = ax.step(power_t0_tend_def.index, power_t0_tend_def.values, where='post', label='Deferred Power')
    # Uncomment if you want to include total power as well
    # ax.step(power_t0_tend_total.index, power_t0_tend_total.values, where='post', label='Total Power')

    # Styling
    ax.set_title(f'{point_name.upper()} - Power Consumption')
    ax.set_xlabel('Time')
    ax.set_ylabel('Power (kW)')
    ax.tick_params(axis='x', rotation=45)

    # Format x-axis to show time more clearly
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    # Collect handles and labels from first subplot only
    if idx == 0:
        all_handles.extend([h1, h2, h3])
        all_labels.extend(['Notification Period', 'Flexibility Window', 'Deferred Power'])

# Add single legend at the bottom
fig.legend(all_handles, all_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01), fontsize=14)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make space for the legend

# Save high-resolution figure
plt.savefig(f'power_consumption_3x3_{SIMULATION}_{SCENARIO}.png', dpi=300)
plt.savefig(f'power_consumption_3x3_{SIMULATION}_{SCENARIO}.pdf')

plt.show()

# Reset matplotlib parameters to default
plt.rcParams.update(plt.rcParamsDefault)

# %%3x3 plot - LARGE FONTS VERSION
# Use seaborn style
sns.set(style="whitegrid")

fig, axs = plt.subplots(3, 3, figsize=(20, 14), sharex=False, sharey=False)

# Define point ordering for 3x3 grid
point_names = ['p1', 'p2', 'p3', 'q1', 'q2', 'q3', 'r1', 'r2', 'r3']

# Collect handles for unified legend
all_handles = []
all_labels = []

# Optional: calculate max power across all to unify y-axis
# max_power = max(max(res_dfs[tuple_point]['df_t0_to_tend_def'].power_kW.values)
#                 for point_name in point_names
#                 for tuple_point in [(points_dates[point_name], latencies[0], fixed_flex_window,
#                                      fixed_delta_notification, fixed_beta, fixed_gamma_buffer)])

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
plt.savefig(f'power_consumption_3x3_LARGE_{SIMULATION}_{SCENARIO}.png', dpi=300)
plt.savefig(f'power_consumption_3x3_LARGE_{SIMULATION}_{SCENARIO}.pdf', dpi=300)
plt.show()

# %%3x3 plot - ALTERNATIVE MEDIUM VERSION
# Use seaborn style
sns.set(style="whitegrid")

fig, axs = plt.subplots(3, 3, figsize=(18, 12), sharex=False, sharey=False)

# Define point ordering for 3x3 grid
point_names = ['p1', 'p2', 'p3', 'q1', 'q2', 'q3', 'r1', 'r2', 'r3']

# Collect handles for unified legend
all_handles = []
all_labels = []

# Optional: calculate max power across all to unify y-axis
# max_power = max(max(res_dfs[tuple_point]['df_t0_to_tend_def'].power_kW.values)
#                 for point_name in point_names
#                 for tuple_point in [(points_dates[point_name], latencies[0], fixed_flex_window,
#                                      fixed_delta_notification, fixed_beta, fixed_gamma_buffer)])

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
                  color=line_color, linewidth=1.8, label='Deferred Power')

    # Formatting
    ax.set_title(f"{point_name.upper()} - Power Consumption", fontweight='bold', fontsize=14)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Power (kW)', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    # ax.set_ylim(0, max_power * 1.05)

    # Date formatting on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

    # Hide top/right spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)

    # Collect legend handles from first plot only
    if idx == 0:
        all_handles.extend([h1, h2, h3])
        all_labels.extend(['Notification Period', 'Flexibility Window', 'Deferred Power'])

# Shared legend at the bottom
fig.legend(all_handles, all_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.03), fontsize=14)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Save + show
# plt.savefig(f'power_consumption_3x3_ALT_{SIMULATION}_{SCENARIO}.png', dpi=300)
plt.show()

# %%TOTAL DC VS AFTER FLEX ACTIVATION
# Access P1 data
tuple_p1 = (points_dates['p1'], latencies[0], fixed_flex_window,
            fixed_delta_notification, fixed_beta, fixed_gamma_buffer)

t_1 = res_dfs[tuple_p1]['t1']
t_2 = res_dfs[tuple_p1]['t2']
t_0 = res_dfs[tuple_p1]['t0']
t_end = res_dfs[tuple_p1]['tend']
LATENCY = res_dfs[tuple_p1]['latency']
delta_notification = res_dfs[tuple_p1]['delta_notification']
gamma_buffer = res_dfs[tuple_p1]['gamma_buffer']
beta = res_dfs[tuple_p1]['beta']

power_def = res_dfs[tuple_p1]['df_t0_to_tend_def'].power_kW
power_total = res_dfs[tuple_p1]['df_t0_to_tend_total'].power_kW

common_index = power_def.index.union(power_total.index)

# Reindex both series
power_def_aligned = power_def.reindex(common_index, method='nearest')
power_total_aligned = power_total.reindex(common_index, method='nearest')

# Subtract
power_diff = power_total_aligned - power_def_aligned

start_point = power_def.index[0]
vertical_line_point = t_1
tend = t_end
vertical_line_point2 = t_2

fig, ax = plt.subplots(figsize=(14, 6))
ax.step(power_total.index, power_total.values, where='post', label='DC Total Power', color='blue', linewidth=2)
ax.step(power_diff.index, power_diff.values, where='post', linestyle='--', color='purple',
        label='DC Power after LAD-Flex', linewidth=2)
# Fill between power_total and power_diff only until t1
ax.fill_between(power_total.index, power_total.values, power_diff.values,
                where=(power_total.index <= vertical_line_point), color='lightblue', alpha=0.5)
# Fill between power_total and power_diff only between t1 and t2
ax.fill_between(power_total.index, power_total.values, power_diff.values,
                where=((power_total.index > vertical_line_point) & (power_total.index <= vertical_line_point2)),
                color='lightgreen', alpha=0.5, label='Flexible energy')
# Fill between power_total and power_diff only after t2
ax.fill_between(power_total.index, power_total.values, power_diff.values,
                where=(power_total.index > vertical_line_point2), color='lightblue', alpha=0.5, label='Rebound energy')

# Add xtick labels
for t, label in zip([t_0, t_1, t_2, t_end], ['$t_0$', '$t_1$', '$t_2$', '$t_{end}$']):
    ax.axvline(t, linestyle='--', color='red', alpha=0.3, linewidth=2)
    ax.text(t, -0.01, label, color="red",
            ha='center', va='top', fontsize=16,
            transform=ax.get_xaxis_transform(), clip_on=False)

# Xticks label size
ax.tick_params(axis='x', labelsize=14)
ax.grid(True)
ax.set_title('P1: LAD-Flex Before and After', fontsize=18)
ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Power (kW)', fontsize=16)
ax.legend(fontsize=14, loc='lower right')
# Show dark spines
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1.2)

plt.tight_layout()
# Save as high resolution png
# plt.savefig(f'LAD_flex_P1_explanation_{SIMULATION}_{SCENARIO}.png', dpi=300)
plt.show()

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
# plt.savefig(f'KDE_deferrable_energy_{SIMULATION}_{SCENARIO}.png', dpi=300)
plt.show()

