import pandas as pd
from utils_uncertainty import *
import numpy as np

# make dataset lighter
N_slice = 100000        # Number of rows to analyze
N_slice_start = 170000  # Starting index

# Define only the columns we need (excluding the ones we would drop)
cols_to_keep = ["task_name", "start_time", "end_time", "wait_time", "runtime_i", "plan_power"]

df_tasks = pd.read_csv("../DATA/df_aa_full_sorted_power.csv", 
                       index_col=0,
                       skiprows=range(1, N_slice_start + 1),  # Skip first 170k data rows
                       nrows=N_slice,  # Read only 100k rows
                       usecols=cols_to_keep,   
                       parse_dates=["start_time", "end_time"])  # Parse dates during read

# Compute t_request
df_tasks["t_request"] = df_tasks["start_time"] - pd.to_timedelta(df_tasks["wait_time"], unit="s")

# Example 1: Perfect information (gamma = 0, no uncertainty)
df_perfect = df_tasks.copy()
df_perfect["duration_predicted"] = df_perfect["runtime_i"]
df_perfect["duration_actual"] = df_perfect["runtime_i"]

# Example 2: Low uncertainty (10% error)
df_gamma_010 = apply_duration_uncertainty(df_tasks, gamma=0.10, random_seed=42)

# Example 3: Medium uncertainty (20% error)
df_gamma_020 = apply_duration_uncertainty(df_tasks, gamma=0.20, random_seed=42)

# Example 4: High uncertainty (30% error)
df_gamma_030 = apply_duration_uncertainty(df_tasks, gamma=0.30, random_seed=42)

# ============================================================================
# VERIFICATION & DIAGNOSTICS
# ============================================================================

def print_uncertainty_stats(df, gamma_label):
    """Print diagnostic statistics about prediction errors"""
    print(f"\n{'='*60}")
    print(f"Prediction Uncertainty Statistics: {gamma_label}")
    print(f"{'='*60}")
    
    if "prediction_error" in df.columns:
        print(f"Mean prediction error: {df['prediction_error'].mean():.2f} seconds")
        print(f"Std dev of error: {df['prediction_error'].std():.2f} seconds")
        print(f"Mean absolute error: {df['prediction_error'].abs().mean():.2f} seconds")
        print(f"Mean percentage error: {df['prediction_error_pct'].mean():.2f}%")
        print(f"Std dev of % error: {df['prediction_error_pct'].std():.2f}%")
        
        # Check how many predictions are significantly off
        high_error = (df['prediction_error'].abs() > 0.5 * df['duration_actual']).sum()
        print(f"\nTasks with |error| > 50% of duration: {high_error} ({100*high_error/len(df):.1f}%)")
        
        # Duration comparison
        print(f"\nOriginal mean duration: {df['duration_actual'].mean():.2f} seconds")
        print(f"Predicted mean duration: {df['duration_predicted'].mean():.2f} seconds")
        
        # Time shift statistics
        time_shift = (df['end_time'] - df['end_time_actual']).dt.total_seconds()
        print(f"\nMean end_time shift: {time_shift.mean():.2f} seconds")
        print(f"Max positive shift: {time_shift.max():.2f} seconds")
        print(f"Max negative shift: {time_shift.min():.2f} seconds")
    else:
        print("Perfect information (no prediction error)")
        print(f"Mean duration: {df['runtime_i'].mean():.2f} seconds")

# Print stats for each scenario
print_uncertainty_stats(df_perfect, "Perfect (γ=0)")
print_uncertainty_stats(df_gamma_010, "Low (γ=0.10)")
print_uncertainty_stats(df_gamma_020, "Medium (γ=0.20)")
print_uncertainty_stats(df_gamma_030, "High (γ=0.30)")

# ============================================================================
# VISUALIZATION: Prediction Error Distribution
# ============================================================================

import matplotlib.pyplot as plt

def plot_prediction_errors(dataframes, labels, colors):
    """Plot prediction error distributions for comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (df, label, color) in enumerate(zip(dataframes, labels, colors)):
        if "prediction_error" not in df.columns:
            continue
            
        ax1 = axes[0, 0] if idx == 0 else axes[0, 1] if idx == 1 else axes[1, 0]
        ax2 = axes[1, 1]
        
        # Individual histogram
        ax1.hist(df['prediction_error_pct'], bins=50, alpha=0.7, color=color, 
                edgecolor='black', density=True)
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax1.set_xlabel('Prediction Error (%)', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Combined KDE
        df['prediction_error_pct'].plot(kind='kde', ax=ax2, label=label, 
                                        color=color, linewidth=2)
    
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Prediction Error (%)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Comparison of Error Distributions', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    # plt.savefig('prediction_error_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot distributions
plot_prediction_errors(
    [df_gamma_010, df_gamma_020, df_gamma_030],
    ['γ=0.10 (10%)', 'γ=0.20 (20%)', 'γ=0.30 (30%)'],
    ['blue', 'orange', 'red']
)

# ============================================================================
# SCATTER PLOT: Actual vs Predicted Duration
# ============================================================================

def plot_actual_vs_predicted(df, gamma_label):
    """Scatter plot comparing actual vs predicted durations"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sample for visualization (plotting all 100k points is slow)
    df_sample = df.sample(n=min(5000, len(df)), random_state=42)
    
    ax.scatter(df_sample['duration_actual'] / 60, 
              df_sample['duration_predicted'] / 60,
              alpha=0.3, s=10, color='steelblue', edgecolor='none')
    
    # Perfect prediction line
    max_dur = max(df_sample['duration_actual'].max(), df_sample['duration_predicted'].max()) / 60
    ax.plot([0, max_dur], [0, max_dur], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Duration (minutes)', fontsize=12)
    ax.set_ylabel('Predicted Duration (minutes)', fontsize=12)
    ax.set_title(f'Actual vs Predicted Task Duration ({gamma_label})', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    # plt.savefig(f'actual_vs_predicted_{gamma_label.replace("=", "").replace(".", "")}.png', 
    #             dpi=300, bbox_inches='tight')
    plt.show()

# Create scatter plots
plot_actual_vs_predicted(df_gamma_020, 'γ=0.20')


# Exporting
df_gamma_010.to_csv("../DATA/df_gamma_010.csv", index=False)
df_gamma_020.to_csv("../DATA/df_gamma_020.csv", index=False)
df_gamma_030.to_csv("../DATA/df_gamma_030.csv", index=False)

# Export choosing the columns so that it matches df_tasks structure
# Reset index to include task_name as a column
cols_to_keep = ["task_name", "start_time", "end_time", "wait_time", "runtime_i", "plan_power"]
df_gamma_010.reset_index()[cols_to_keep].to_csv("../DATA/df_tasks_gamma_010.csv", index=False)
df_gamma_020.reset_index()[cols_to_keep].to_csv("../DATA/df_tasks_gamma_020.csv", index=False)
df_gamma_030.reset_index()[cols_to_keep].to_csv("../DATA/df_tasks_gamma_030.csv", index=False)
