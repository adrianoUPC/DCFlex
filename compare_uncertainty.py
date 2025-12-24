"""
compare_uncertainty.py

This script compares the impact of task duration uncertainty (zeta parameter) on
LAD-Flex's task deferability classification. It creates confusion matrices comparing
zeta scenarios (0.10, 0.20, 0.30) against the baseline (zeta=0) by aggregating
decisions across all parameter combinations.

Inputs:
    - EXPORTS/res_dfs_DEFAULT_BASELINE_ZETA_0.joblib
    - EXPORTS/res_dfs_DEFAULT_BASELINE_ZETA_010.joblib
    - EXPORTS/res_dfs_DEFAULT_BASELINE_ZETA_020.joblib
    - EXPORTS/res_dfs_DEFAULT_BASELINE_ZETA_030.joblib

Outputs:
    - PLOTS/confusion_matrix_zeta_comparison_DEFAULT_BASELINE.png
    - PLOTS/confusion_matrix_zeta_comparison_DEFAULT_BASELINE.pdf
    - PLOTS/metrics_trend_zeta_DEFAULT_BASELINE.png (optional)
    - PLOTS/metrics_trend_zeta_DEFAULT_BASELINE.pdf (optional)
    - EXPORTS/metrics_zeta_comparison_DEFAULT_BASELINE.csv
    - EXPORTS/confusion_matrix_counts_DEFAULT_BASELINE.csv

Author: LAD-Flex Research Team
Date: 2025-12-23
"""

# %% Imports and Configuration
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns

SIMULATION = "DEFAULT"
SCENARIO = "BASELINE"
ZETA = [0, 0.10, 0.30, 0.50]

# %% Load Data
print("="*70)
print("Loading zeta scenario data...")
print("="*70)

# Zeta suffix mapping (following import_DEFAULT.py pattern)
zeta_suffix_map = {
    0: "ZETA_0",
    0.10: "ZETA_010",
    0.20: "ZETA_020",
    0.30: "ZETA_030",
    0.50: "ZETA_050",
    0.70: "ZETA_070"
}

# Load all res_dfs dictionaries
res_dfs_by_zeta = {}
for zeta in ZETA:
    suffix = zeta_suffix_map[zeta]
    filename = f"EXPORTS/res_dfs_{SIMULATION}_{SCENARIO}_{suffix}.joblib"
    res_dfs_by_zeta[zeta] = joblib.load(filename)
    print(f"Loaded zeta={zeta}: {len(res_dfs_by_zeta[zeta])} parameter combinations")

print()

# %% Extract Tasks

def extract_all_tasks(res_dfs_dict):
    """
    Extract all tasks from all parameter combinations.

    Args:
        res_dfs_dict: Dictionary with results for all parameter combinations

    Returns:
        tuple: (deferrable_tasks, not_deferrable_tasks)
            - deferrable_tasks: set of (task_name, start_time) tuples
            - not_deferrable_tasks: set of (task_name, start_time) tuples
    """
    deferrable_tasks = set()
    not_deferrable_tasks = set()

    for key, value in res_dfs_dict.items():
        # Extract deferrable tasks
        df_def = value['df_deferrable']
        if len(df_def) > 0:
            for task_name, row in df_def.iterrows():
                # Use task_name (index) and start_time as unique identifier
                task_id = (task_name, row['start_time'])
                deferrable_tasks.add(task_id)

        # Extract non-deferrable tasks
        df_not_def = value['df_not_deferrable']
        if len(df_not_def) > 0:
            for task_name, row in df_not_def.iterrows():
                # Use task_name (index) and start_time as unique identifier
                task_id = (task_name, row['start_time'])
                not_deferrable_tasks.add(task_id)

    return deferrable_tasks, not_deferrable_tasks


print("Extracting tasks across all scenarios...")
print("-"*70)

# Extract for each gamma
task_sets = {}
for zeta in GAMMA:
    deferrable, not_deferrable = extract_all_tasks(res_dfs_by_zeta[gamma])
    all_tasks = deferrable | not_deferrable
    task_sets[gamma] = {
        'deferrable': deferrable,
        'not_deferrable': not_deferrable,
        'all_tasks': all_tasks
    }
    print(f"zeta {gamma}: {len(deferrable)} deferrable, {len(not_deferrable)} not deferrable, {len(all_tasks)} total tasks")

print()

# %% Build Confusion Matrices

def build_confusion_matrix(baseline_tasks, comparison_tasks):
    """
    Build confusion matrix comparing zeta scenario against baseline.

    Args:
        baseline_tasks: dict with 'deferrable', 'not_deferrable', 'all_tasks'
        comparison_tasks: dict with 'deferrable', 'not_deferrable', 'all_tasks'

    Returns:
        dict with TP, FP, FN, TN counts and task lists
    """
    # Get all unique tasks from both scenarios
    all_tasks = baseline_tasks['all_tasks'] | comparison_tasks['all_tasks']

    baseline_def = baseline_tasks['deferrable']
    comparison_def = comparison_tasks['deferrable']

    # Calculate confusion matrix components
    TP = baseline_def & comparison_def  # Both say deferrable
    FP = comparison_def - baseline_def   # Comparison says deferrable, baseline says no
    FN = baseline_def - comparison_def   # Baseline says deferrable, comparison says no
    TN = (all_tasks - baseline_def) & (all_tasks - comparison_def)  # Both say not deferrable

    return {
        'TP': len(TP), 'FP': len(FP), 'FN': len(FN), 'TN': len(TN),
        'TP_tasks': TP, 'FP_tasks': FP, 'FN_tasks': FN, 'TN_tasks': TN
    }


print("Building confusion matrices...")
print("-"*70)

# Build confusion matrices
confusion_matrices = {}
baseline_tasks = task_sets[0]  # zeta = 0 is baseline

for zeta in [0.10, 0.30, 0.50]:
    cm = build_confusion_matrix(baseline_tasks, task_sets[gamma])
    confusion_matrices[gamma] = cm
    print(f"zeta {gamma} vs 0: TP={cm['TP']}, FP={cm['FP']}, FN={cm['FN']}, TN={cm['TN']}")

    # Validation: check that TP + FP + FN + TN equals total tasks
    total = cm['TP'] + cm['FP'] + cm['FN'] + cm['TN']
    all_tasks_count = len(baseline_tasks['all_tasks'] | task_sets[gamma]['all_tasks'])
    assert total == all_tasks_count, f"Confusion matrix sum mismatch: {total} != {all_tasks_count}"

print()

# %% Calculate Metrics

def calculate_metrics(cm_dict):
    """
    Calculate classification metrics from confusion matrix.

    Args:
        cm_dict: Dictionary with TP, FP, FN, TN counts

    Returns:
        Dictionary with accuracy, precision, recall, f1_score, specificity
    """
    TP, FP, FN, TN = cm_dict['TP'], cm_dict['FP'], cm_dict['FN'], cm_dict['TN']

    # Accuracy: (TP + TN) / Total
    total = TP + FP + FN + TN
    accuracy = (TP + TN) / total if total > 0 else 0

    # Precision: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Recall (Sensitivity): TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Specificity: TN / (TN + FP) - useful for understanding TN performance
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity
    }


# Calculate metrics for all comparisons
metrics = {}
for zeta in [0.10, 0.30, 0.50]:
    metrics[gamma] = calculate_metrics(confusion_matrices[gamma])

# %% Visualize Confusion Matrices

# Use seaborn style
sns.set(style="whitegrid")

# Create figure with 1x3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes = axes.flatten()

# Color scheme - colorblind friendly (sequential Blues)
cmap = sns.color_palette("Blues", as_cmap=True)

gamma_list = [0.10, 0.30, 0.50]

for idx, zeta in enumerate(gamma_list):
    ax = axes[idx]
    cm = confusion_matrices[gamma]
    met = metrics[gamma]

    # Create 2x2 confusion matrix array
    cm_array = np.array([
        [cm['TP'], cm['FN']],  # Row: Baseline = Deferrable
        [cm['FP'], cm['TN']]   # Row: Baseline = Not Deferrable
    ])

    # Create labels with counts and percentages
    total = cm_array.sum()
    annot_array = np.array([
        [f"{cm['TP']}\n({100*cm['TP']/total:.1f}%)",
         f"{cm['FN']}\n({100*cm['FN']/total:.1f}%)"],
        [f"{cm['FP']}\n({100*cm['FP']/total:.1f}%)",
         f"{cm['TN']}\n({100*cm['TN']/total:.1f}%)"]
    ])

    # Plot heatmap
    sns.heatmap(cm_array, annot=annot_array, fmt='', cmap=cmap,
                cbar=True, ax=ax, linewidths=2, linecolor='black',
                vmin=0, vmax=cm_array.max())

    # Set labels
    ax.set_xlabel(f'Predicted (γ = {gamma:.2f})', fontsize=14)
    ax.set_ylabel('Actual (γ = 0, Baseline)', fontsize=14)
    ax.set_title(f'Confusion Matrix: γ = {gamma} vs γ = 0\n' +
                 f'Acc: {met["accuracy"]:.3f} | Prec: {met["precision"]:.3f} | ' +
                 f'Rec: {met["recall"]:.3f} | F1: {met["f1_score"]:.3f}',
                 fontsize=13, fontweight='bold')

    # Set tick labels
    ax.set_xticklabels(['Deferrable', 'Not Deferrable'], rotation=0, fontsize=12)
    ax.set_yticklabels(['Deferrable', 'Not Deferrable'], rotation=0, fontsize=12)

# Overall title
fig.suptitle('Task Deferability Classification: Impact of Duration Uncertainty (γ)',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()

# Save figures
plt.savefig(f'PLOTS/confusion_matrix_zeta_comparison_{SIMULATION}_{SCENARIO}.png',
            dpi=300, bbox_inches='tight')
plt.savefig(f'PLOTS/confusion_matrix_zeta_comparison_{SIMULATION}_{SCENARIO}.pdf',
            bbox_inches='tight')

print("Confusion matrix plots saved:")
print(f"  - PLOTS/confusion_matrix_zeta_comparison_{SIMULATION}_{SCENARIO}.png")
print(f"  - PLOTS/confusion_matrix_zeta_comparison_{SIMULATION}_{SCENARIO}.pdf")
print()

plt.show()

# Reset to default
sns.reset_defaults()

# %% Confusion Matrices for Sample Week Only

print("="*70)
print("Building confusion matrices for SAMPLE WEEK only...")
print("="*70)

# Define sample week (same as in import_DEFAULT.py)
sample_week = pd.date_range(start='1970-02-01', periods=7, freq='D')
start_date = sample_week[0]
end_date = sample_week[-1] + pd.Timedelta(days=1)  # Include the last day

def extract_tasks_for_date_range(res_dfs_dict, start_date, end_date):
    """
    Extract tasks from a specific date range.
    
    Args:
        res_dfs_dict: Dictionary with results for all parameter combinations
        start_date: Start of date range
        end_date: End of date range (exclusive)
    
    Returns:
        tuple: (deferrable_tasks, not_deferrable_tasks)
    """
    deferrable_tasks = set()
    not_deferrable_tasks = set()
    
    for key, value in res_dfs_dict.items():
        # key is (date, latency, flex_window, delta_notification, beta, gamma_buffer)
        task_date = key[0]
        
        # Filter by date range
        if start_date <= task_date < end_date:
            # Extract deferrable tasks
            df_def = value['df_deferrable']
            if len(df_def) > 0:
                for task_name, row in df_def.iterrows():
                    task_id = (task_name, row['start_time'])
                    deferrable_tasks.add(task_id)
            
            # Extract non-deferrable tasks
            df_not_def = value['df_not_deferrable']
            if len(df_not_def) > 0:
                for task_name, row in df_not_def.iterrows():
                    task_id = (task_name, row['start_time'])
                    not_deferrable_tasks.add(task_id)
    
    return deferrable_tasks, not_deferrable_tasks

# Extract tasks for sample week
task_sets_week = {}
for zeta in GAMMA:
    deferrable, not_deferrable = extract_tasks_for_date_range(
        res_dfs_by_zeta[gamma], start_date, end_date
    )
    all_tasks = deferrable | not_deferrable
    task_sets_week[gamma] = {
        'deferrable': deferrable,
        'not_deferrable': not_deferrable,
        'all_tasks': all_tasks
    }
    print(f"zeta {gamma} (sample week): {len(deferrable)} deferrable, {len(not_deferrable)} not deferrable, {len(all_tasks)} total tasks")

print()

# Build confusion matrices for sample week
confusion_matrices_week = {}
baseline_tasks_week = task_sets_week[0]

for zeta in [0.10, 0.30, 0.50]:
    cm = build_confusion_matrix(baseline_tasks_week, task_sets_week[gamma])
    confusion_matrices_week[gamma] = cm
    print(f"Sample Week - zeta {gamma} vs 0: TP={cm['TP']}, FP={cm['FP']}, FN={cm['FN']}, TN={cm['TN']}")

print()

# Calculate metrics for sample week
metrics_week = {}
for zeta in [0.10, 0.30, 0.50]:
    metrics_week[gamma] = calculate_metrics(confusion_matrices_week[gamma])

# %% Visualize Sample Week Confusion Matrices

sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes = axes.flatten()

cmap = sns.color_palette("Blues", as_cmap=True)

gamma_list = [0.10, 0.30, 0.50]

for idx, zeta in enumerate(gamma_list):
    ax = axes[idx]
    cm = confusion_matrices_week[gamma]
    met = metrics_week[gamma]

    # Create 2x2 confusion matrix array
    cm_array = np.array([
        [cm['TP'], cm['FN']],
        [cm['FP'], cm['TN']]
    ])

    # Create labels with counts and percentages
    total = cm_array.sum()
    annot_array = np.array([
        [f"{cm['TP']}\n({100*cm['TP']/total:.1f}%)",
         f"{cm['FN']}\n({100*cm['FN']/total:.1f}%)"],
        [f"{cm['FP']}\n({100*cm['FP']/total:.1f}%)",
         f"{cm['TN']}\n({100*cm['TN']/total:.1f}%)"]
    ])

    # Plot heatmap
    sns.heatmap(cm_array, annot=annot_array, fmt='', cmap=cmap,
                cbar=True, ax=ax, linewidths=2, linecolor='black',
                vmin=0, vmax=cm_array.max())

    # Set labels
    ax.set_xlabel(f'Predicted (γ = {gamma:.2f})', fontsize=14)
    ax.set_ylabel('Actual (γ = 0, Baseline)', fontsize=14)
    ax.set_title(f'Confusion Matrix: γ = {gamma} vs γ = 0\n' +
                 f'Acc: {met["accuracy"]:.3f} | Prec: {met["precision"]:.3f} | ' +
                 f'Rec: {met["recall"]:.3f} | F1: {met["f1_score"]:.3f}',
                 fontsize=13, fontweight='bold')

    # Set tick labels
    ax.set_xticklabels(['Deferrable', 'Not Deferrable'], rotation=0, fontsize=12)
    ax.set_yticklabels(['Deferrable', 'Not Deferrable'], rotation=0, fontsize=12)

# Overall title
fig.suptitle('Task Deferability Classification (Sample Week): Impact of Duration Uncertainty (γ)',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

# Save figures
plt.savefig(f'PLOTS/confusion_matrix_zeta_comparison_SAMPLE_WEEK_{SIMULATION}_{SCENARIO}.png',
            dpi=300, bbox_inches='tight')
plt.savefig(f'PLOTS/confusion_matrix_zeta_comparison_SAMPLE_WEEK_{SIMULATION}_{SCENARIO}.pdf',
            bbox_inches='tight')

print("Sample week confusion matrix plots saved:")
print(f"  - PLOTS/confusion_matrix_zeta_comparison_SAMPLE_WEEK_{SIMULATION}_{SCENARIO}.png")
print(f"  - PLOTS/confusion_matrix_zeta_comparison_SAMPLE_WEEK_{SIMULATION}_{SCENARIO}.pdf")
print()

plt.show()

# Save sample week metrics
metrics_week_df = pd.DataFrame(metrics_week).T
metrics_week_df.index.name = 'Gamma'
metrics_week_df = metrics_week_df.round(4)

print("Sample Week Classification Metrics Summary:")
print("="*70)
print(metrics_week_df.to_string())
print("="*70)
print()

metrics_week_df.to_csv(f'EXPORTS/metrics_zeta_comparison_SAMPLE_WEEK_{SIMULATION}_{SCENARIO}.csv')
print(f"Sample week metrics saved to EXPORTS/metrics_zeta_comparison_SAMPLE_WEEK_{SIMULATION}_{SCENARIO}.csv")
print()

sns.reset_defaults()

# %% Metrics Summary (Full Period)

# Create metrics summary DataFrame
metrics_df = pd.DataFrame(metrics).T
metrics_df.index.name = 'Gamma'
metrics_df = metrics_df.round(4)

print("Classification Metrics Summary:")
print("="*70)
print(metrics_df.to_string())
print("="*70)
print()

# Save metrics
metrics_df.to_csv(f'EXPORTS/metrics_zeta_comparison_{SIMULATION}_{SCENARIO}.csv')
print(f"Metrics saved to EXPORTS/metrics_zeta_comparison_{SIMULATION}_{SCENARIO}.csv")
print()

# Create confusion matrix counts summary
cm_summary = pd.DataFrame({
    gamma: {
        'TP': confusion_matrices[gamma]['TP'],
        'FP': confusion_matrices[gamma]['FP'],
        'FN': confusion_matrices[gamma]['FN'],
        'TN': confusion_matrices[gamma]['TN'],
        'Total': sum([confusion_matrices[gamma][k] for k in ['TP', 'FP', 'FN', 'TN']])
    }
    for zeta in [0.10, 0.30, 0.50]
}).T

cm_summary.index.name = 'Gamma'

print("Confusion Matrix Counts:")
print("="*70)
print(cm_summary.to_string())
print("="*70)
print()

cm_summary.to_csv(f'EXPORTS/confusion_matrix_counts_{SIMULATION}_{SCENARIO}.csv')
print(f"Confusion matrix counts saved to EXPORTS/confusion_matrix_counts_{SIMULATION}_{SCENARIO}.csv")
print()

# %% Additional Analysis: Metrics Trend Plot

# Plot metrics trends across zeta values
fig, ax = plt.subplots(figsize=(12, 6))

gamma_vals = [0.10, 0.30, 0.50]
accuracy_vals = [metrics[g]['accuracy'] for g in gamma_vals]
precision_vals = [metrics[g]['precision'] for g in gamma_vals]
recall_vals = [metrics[g]['recall'] for g in gamma_vals]
f1_vals = [metrics[g]['f1_score'] for g in gamma_vals]

ax.plot(gamma_vals, accuracy_vals, marker='o', linewidth=2, markersize=8, label='Accuracy', color='#0571B0')
ax.plot(gamma_vals, precision_vals, marker='s', linewidth=2, markersize=8, label='Precision', color='#FC9272')
ax.plot(gamma_vals, recall_vals, marker='^', linewidth=2, markersize=8, label='Recall', color='#A6D854')
ax.plot(gamma_vals, f1_vals, marker='d', linewidth=2, markersize=8, label='F1-Score', color='#E7298A')

ax.set_xlabel('Task Duration Uncertainty (γ)', fontsize=14)
ax.set_ylabel('Metric Value', fontsize=14)
ax.set_title('Classification Metrics vs Duration Uncertainty', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
ax.set_xticks(gamma_vals)

plt.tight_layout()
plt.savefig(f'PLOTS/metrics_trend_zeta_{SIMULATION}_{SCENARIO}.png', dpi=300)
plt.savefig(f'PLOTS/metrics_trend_zeta_{SIMULATION}_{SCENARIO}.pdf')

print("Metrics trend plots saved:")
print(f"  - PLOTS/metrics_trend_zeta_{SIMULATION}_{SCENARIO}.png")
print(f"  - PLOTS/metrics_trend_zeta_{SIMULATION}_{SCENARIO}.pdf")
print()

plt.show()

# %% Deferred Power Time Series: 3x3 Grid Comparison Across zeta Values

print("="*70)
print("Creating 3x3 deferred power time series comparison...")
print("="*70)

# Load notable points from baseline (zeta=0)
points_dates = joblib.load(f"EXPORTS/points_dates_{SIMULATION}_{SCENARIO}_GAMMA_0.joblib")

# Get parameters (assuming same across all zeta scenarios)
from get_parameters import return_params
params = return_params(SIMULATION, SCENARIO)
latencies = params['latencies']
flex_windows = params['flex_windows']
delta_notifications = params['delta_notifications']
betas = params['betas']
gamma_buffers = params['gamma_buffers']

# Use first values as fixed parameters
fixed_flex_window = flex_windows[0]
fixed_delta_notification = delta_notifications[0]
fixed_beta = betas[0]
fixed_gamma_buffer = gamma_buffers[0]

# Create 3x3 subplot grid
fig, axs = plt.subplots(3, 3, figsize=(20, 14), sharex=False, sharey=False)

# Define point ordering for 3x3 grid
point_names = ['p1', 'p2', 'p3', 'q1', 'q2', 'q3', 'r1', 'r2', 'r3']

# Color palette for different zeta values
colors = {
    0: '#2E86AB',      # blue
    0.10: '#A23B72',   # purple
    0.30: '#F18F01',   # orange
    0.50: '#C73E1D'    # red
}

all_handles = []
all_labels = []

# Plot each point
for idx, point_name in enumerate(point_names):
    row, col = divmod(idx, 3)
    ax = axs[row, col]
    
    # Plot power curves for each zeta value
    for zeta in sorted(zeta):
        res_dfs = res_dfs_by_zeta[gamma]
        
        # Get data for this point
        tuple_point = (points_dates[point_name], latencies[0], fixed_flex_window,
                       fixed_delta_notification, fixed_beta, fixed_gamma_buffer)
        
        if tuple_point in res_dfs:
            t0 = res_dfs[tuple_point]['t0']
            t1 = res_dfs[tuple_point]['t1']
            t2 = res_dfs[tuple_point]['t2']
            tend = res_dfs[tuple_point]['tend']
            
            power_t0_tend_def = res_dfs[tuple_point]['df_t0_to_tend_def'].power_kW
            
            # Plot power curve
            h, = ax.step(power_t0_tend_def.index, power_t0_tend_def.values, 
                        where='post', linewidth=2, alpha=0.8,
                        color=colors[gamma], label=f'γ={gamma}')
            
            # Add shaded regions only on the first zeta (to avoid overlapping)
            if zeta == 0:
                ax.axvspan(t0, t1, color='green', alpha=0.15)
                ax.axvspan(t1, t2, color='red', alpha=0.15)
    
    # Styling
    ax.set_title(f"{point_name.upper()} - {points_dates[point_name].date()}", 
                fontweight='bold', fontsize=16)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Power (kW)', fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    # Collect legend handles from first subplot only
    if idx == 0:
        handles, labels = ax.get_legend_handles_labels()
        all_handles = handles
        all_labels = labels

# Add single legend at the bottom
fig.legend(all_handles, all_labels, loc='lower center', ncol=len(zeta), 
          bbox_to_anchor=(0.5, 0.015), fontsize=20)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.12, top=0.97)

# Save figures
plt.savefig(f'PLOTS/deferred_power_timeseries_3x3_gamma_{SIMULATION}_{SCENARIO}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'PLOTS/deferred_power_timeseries_3x3_gamma_{SIMULATION}_{SCENARIO}.pdf', bbox_inches='tight')

print("Deferred power time series plots saved:")
print(f"  - PLOTS/deferred_power_timeseries_3x3_gamma_{SIMULATION}_{SCENARIO}.png")
print(f"  - PLOTS/deferred_power_timeseries_3x3_gamma_{SIMULATION}_{SCENARIO}.pdf")
print()

plt.show()

print("="*70)
print("Analysis complete!")
print("="*70)
# %%

