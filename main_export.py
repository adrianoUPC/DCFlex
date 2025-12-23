import pandas as pd
import time
from utils import *
from get_parameters import *
import joblib

# set up timer for performance measurement
start_time = time.time()
print("Start time: ", start_time)

# make dataset lighter
N_slice = 100000        # Number of rows to analyze
N_slice_start = 170000  # Starting index

# Define only the columns we need (excluding the ones we would drop)
cols_to_keep = ["task_name", "start_time", "end_time", "wait_time", "runtime_i", "plan_power"]


# OBS: we can use this to get results for the duration uncertainty scenarios too (but only if the columns are the same)
GAMMA = 0.70

# Format GAMMA suffix for filenames
if GAMMA == 0:
    filename = "DATA/df_aa_full_sorted_power.csv"
    gamma_suffix = "GAMMA_0"
    df_tasks = pd.read_csv(filename, 
                       index_col=0,
                       skiprows=range(1, N_slice_start + 1),  # Skip first 170k data rows
                       nrows=N_slice,  # Read only 100k rows
                       usecols=cols_to_keep,   
                       parse_dates=["start_time", "end_time"])  # Parse dates during read

elif GAMMA == 0.10:
    filename = "DATA/df_tasks_gamma_010.csv"
    gamma_suffix = "GAMMA_010"
elif GAMMA == 0.20:
    filename = "DATA/df_tasks_gamma_020.csv"
    gamma_suffix = "GAMMA_020"
elif GAMMA == 0.30:
    filename = "DATA/df_tasks_gamma_030.csv"
    gamma_suffix = "GAMMA_030"
elif GAMMA == 0.50:
    filename = "DATA/df_tasks_gamma_050.csv"
    gamma_suffix = "GAMMA_050"
elif GAMMA == 0.70:
    filename = "DATA/df_tasks_gamma_070.csv"
    gamma_suffix = "GAMMA_070"
    
df_tasks = pd.read_csv(filename, 
                       index_col=0,
                    #    skiprows=range(1, N_slice_start + 1),  # Skip first 170k data rows
                    #    nrows=N_slice,  # Read only 100k rows
                       usecols=cols_to_keep,   
                       parse_dates=["start_time", "end_time"])  # Parse dates during read

# Compute t_request
df_tasks["t_request"] = df_tasks["start_time"] - pd.to_timedelta(df_tasks["wait_time"], unit="s")

# %% Applying the LAD-FLEX algorithm and exporting results
# TODO: to integrate task duration uncertainty we need the same columns as df_tasks

# Compute timeline and power series
timeline = get_timeline(df_tasks)
df_power_energy = fast_power_energy_series(df_tasks, timeline)

SIMULATION = "DEFAULT" # choices: "DEFAULT", "SENSITIVE", "OPT"
# SCENARIO = "BASELINE"
SCENARIO = "BASELINE" # choices: "BASELINE", "mFRR", "DAM"

if SIMULATION == "DEFAULT":
    #consider the dates for the full time period (only for the default simulation)
    dates = pd.date_range(start=df_tasks['t_request'].min(), end=df_tasks['t_request'].max(), freq='H')
    params = return_params(SIMULATION, SCENARIO)
    latencies = params['latencies']
    flex_windows = params['flex_windows']
    delta_notifications = params['delta_notifications']
    betas = params['betas']
    gamma_buffers = params['gamma_buffers']
    
#immediately filter out tasks that have an exceedingly large runtime
max_runtime = max(delta_notifications) + max(flex_windows) + max(gamma_buffers)
max_runtime_seconds = max_runtime.total_seconds()
df_tasks = df_tasks[df_tasks['runtime_i'] <= max_runtime_seconds].copy()

#initialize dictionaries to store results
res_optimization = {}
res_dfs = {}
summary_data = []

# Start the simulation
for latency in latencies:
    for flex_window in flex_windows:
        for beta in betas:
            for gamma_buffer in gamma_buffers:
                for delta_notification in delta_notifications:
                    for date in dates:    
                        # Initialize res_optimization if needed
                        if date not in res_optimization:
                            res_optimization[date] = {
                                'delta': [], 'F': [], 'V': [], 'R': []
                            }
                        
                        # Define time windows
                        t1 = date
                        t2 = date + flex_window
                        t0 = t1 - delta_notification
                        tend = t2 + gamma_buffer

                        # Tuple key for this combination (includes beta and gamma_buffer)
                        key = (date, latency, flex_window, delta_notification, beta, gamma_buffer)
                        #run LAD-FLEX
                        df_deferrable, df_not_deferrable = LAD_flex(df_tasks, latency,
                            t1, t2, beta, delta_notification, gamma_buffer)
                        
                        # Compute power/energy time series
                        timeline_deferrable = get_timeline(df_deferrable)
                        df_deferrable_power_energy = fast_power_energy_series(df_deferrable, timeline_deferrable)

                        # Extract energy in flexibility window
                        energy_window = df_deferrable_power_energy[
                            (df_deferrable_power_energy.index >= t1) & (df_deferrable_power_energy.index < t2)]

                        df_total_power_energy = df_power_energy[
                            (df_power_energy.index >= t1) & (df_power_energy.index < t2)]

                        # Compute extended time series (t0 to tend)
                        df_t0_to_tend_def = df_deferrable_power_energy[(
                                (df_deferrable_power_energy.index >= t0) & (df_deferrable_power_energy.index < tend))
                        ]

                        df_t0_to_tend_total = df_power_energy[(
                                (df_power_energy.index >= t0) & (df_power_energy.index < tend))
                        ]

                        # Compute rebound energy
                        rebound_mask = (
                                ((df_power_energy.index >= t0) & (df_power_energy.index < t1)) |
                                ((df_power_energy.index >= t2) & (df_power_energy.index < tend))
                        )
                        rebound_df = df_power_energy[rebound_mask]

                        # Compute KPIs
                        deferrable_energy_sum = energy_window['energy_kWh'].sum()
                        total_energy_sum = df_total_power_energy['energy_kWh'].sum()
                        percentage_flexible = (deferrable_energy_sum / total_energy_sum * 100) if total_energy_sum != 0 else 0
                        rebound_energy_sum = rebound_df['energy_kWh'].sum()

                                        # Store detailed results (DataFrames + metadata, NO scalar KPIs)
                        res_dfs[key] = {
                            # Metadata (flattened)
                            't1': t1,
                            't2': t2,
                            't0': t0,
                            'tend': tend,
                            'latency': latency,
                            'beta': beta,
                            'delta_notification': delta_notification,
                            'gamma_buffer': gamma_buffer,

                            # DataFrames only
                            'df_deferrable': df_deferrable,
                            'df_not_deferrable': df_not_deferrable,
                            'df_deferrable_power_energy': df_deferrable_power_energy,
                            'deferrable_energy_window': energy_window,
                            'df_total_power_energy': df_total_power_energy,
                            'df_t0_to_tend_def': df_t0_to_tend_def,
                            'df_t0_to_tend_total': df_t0_to_tend_total,
                            'df_rebound_energy': rebound_df
                        }

                        # Store summary KPIs separately (will go into df_summary)
                        summary_data.append({
                            'timestamp': date,
                            'latency': latency,
                            'flex_window': flex_window,
                            'delta_notification': delta_notification,
                            'beta': beta,  # ✅ Beta parameter for sensitivity analysis
                            'gamma_buffer': gamma_buffer,  # ✅ NEW: Gamma_buffer parameter for sensitivity analysis
                            'deferrable_energy_kWh': deferrable_energy_sum,
                            'total_energy_kWh': total_energy_sum,
                            'percentage_flexible_energy': percentage_flexible,
                            'deferrable_task_count': len(df_deferrable),
                            'rebound_energy_kWh': rebound_energy_sum,
                        })

                        # Optimization data (OPT mode only)
                        if latency == 600 and flex_window == pd.Timedelta(hours=3):
                            F = deferrable_energy_sum
                            V = Value(delta_notification)
                            R = F * V

                            res_optimization[date]['delta'].append(delta_notification)
                            res_optimization[date]['F'].append(F)
                            res_optimization[date]['V'].append(V)
                            res_optimization[date]['R'].append(R)

# create df_summary and set multi-index
df_summary = pd.DataFrame(summary_data)
df_summary.set_index(['timestamp', 'latency', 'flex_window', 'delta_notification', 'beta', 'gamma_buffer'], inplace=True)

joblib.dump(df_summary, f'EXPORTS/df_summary_{SIMULATION}_{SCENARIO}_{gamma_suffix}.joblib')
joblib.dump(res_dfs, f'EXPORTS/res_dfs_{SIMULATION}_{SCENARIO}_{gamma_suffix}.joblib')

checkpoint_time = time.time()
print("Checkpoint time: ", checkpoint_time)

end_time = time.time()
print("End time: ", end_time)
# %%
