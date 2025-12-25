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

# Example 1: Perfect information (zeta = 0, no uncertainty)
df_perfect = df_tasks.copy()
df_perfect["duration_predicted"] = df_perfect["runtime_i"]
df_perfect["duration_actual"] = df_perfect["runtime_i"]

# Example 2: Low uncertainty (10% error)
df_zeta_010 = apply_duration_uncertainty(df_tasks, zeta=0.10, random_seed=42)

# Example 3: Medium uncertainty (20% error)
df_zeta_020 = apply_duration_uncertainty(df_tasks, zeta=0.20, random_seed=42)

# Example 4: High uncertainty (30% error)
df_zeta_030 = apply_duration_uncertainty(df_tasks, zeta=0.30, random_seed=42)

# Example 5: Higher uncertainty (50% error)
df_zeta_050 = apply_duration_uncertainty(df_tasks, zeta=0.50, random_seed=42)

# Example 6: Very high uncertainty (70% error)
df_zeta_070 = apply_duration_uncertainty(df_tasks, zeta=0.70, random_seed=42)


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
    [df_zeta_010, df_zeta_020, df_zeta_030],
    ['ζ=0.10 (10%)', 'ζ=0.20 (20%)', 'ζ=0.30 (30%)'],
    ['blue', 'orange', 'red']
)

# ============================================================================
# SCATTER PLOT: Actual vs Predicted Duration
# ============================================================================

def plot_actual_vs_predicted(df, zeta_label):
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
    ax.set_title(f'Actual vs Predicted Task Duration ({zeta_label})', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    # plt.savefig(f'actual_vs_predicted_{zeta_label.replace("=", "").replace(".", "")}.png', 
    #             dpi=300, bbox_inches='tight')
    plt.show()

# Create comprehensive comparison plot
zeta_dataframes = {
    0.10: df_zeta_010,
    0.20: df_zeta_020,
    0.30: df_zeta_030,
    0.50: df_zeta_050,
    0.70: df_zeta_070
}


# # Exporting
 # df_zeta_010.to_csv("../DATA/df_zeta_010.csv", index=False)
 # df_zeta_020.to_csv("../DATA/df_zeta_020.csv", index=False)
 # df_zeta_030.to_csv("../DATA/df_zeta_030.csv", index=False)
 # df_zeta_050.to_csv("../DATA/df_zeta_050.csv", index=False)
 # df_zeta_070.to_csv("../DATA/df_zeta_070.csv", index=False)

# # Export choosing the columns so that it matches df_tasks structure
# # Reset index to include task_name as a column
# cols_to_keep = ["task_name", "start_time", "end_time", "wait_time", "runtime_i", "plan_power"]
# df_zeta_010.reset_index()[cols_to_keep].to_csv("../DATA/df_tasks_zeta_010.csv", index=False)
# df_zeta_020.reset_index()[cols_to_keep].to_csv("../DATA/df_tasks_zeta_020.csv", index=False)
# df_zeta_030.reset_index()[cols_to_keep].to_csv("../DATA/df_tasks_zeta_030.csv", index=False)
# df_zeta_050.reset_index()[cols_to_keep].to_csv("../DATA/df_tasks_zeta_050.csv", index=False)
# df_zeta_070.reset_index()[cols_to_keep].to_csv("../DATA/df_tasks_zeta_070.csv", index=False)

# ============================================================================
# STANDALONE PLOT: Comparison of Error Distributions (γ=0.1, 0.3, 0.5)
# ============================================================================

def plot_error_distributions_standalone(df_010, df_030, df_050, save_path=None):
    """
    Create a standalone KDE comparison plot for ζ=0.1, 0.3, and 0.5.
    
    Parameters:
    -----------
    df_010 : pd.DataFrame
        Dataframe with zeta=0.1 uncertainty
    df_030 : pd.DataFrame
        Dataframe with zeta=0.3 uncertainty
    df_050 : pd.DataFrame
        Dataframe with zeta=0.5 uncertainty
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot KDE for each zeta value
    df_010['prediction_error_pct'].plot(kind='kde', ax=ax, label='ζ=0.10', 
                                        color='blue', linewidth=2.5, alpha=0.8)
    df_030['prediction_error_pct'].plot(kind='kde', ax=ax, label='ζ=0.30', 
                                        color='orange', linewidth=2.5, alpha=0.8)
    df_050['prediction_error_pct'].plot(kind='kde', ax=ax, label='ζ=0.50', 
                                        color='red', linewidth=2.5, alpha=0.8)
    
    # Add zero error reference line
    ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Zero Error')
    
    # Labels and formatting
    ax.set_xlabel('Prediction Error (%)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Comparison of Error Distributions', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)
    ax.set_xlim(-150, 150)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Create the standalone plot
plot_error_distributions_standalone(
    df_zeta_010,
    df_zeta_030,
    df_zeta_050,
    save_path='../PLOTS/error_distributions_zeta_010_030_050.png'
)

# ============================================================================
# COMBINED PLOT: Error Distributions + Scatter Plot (ζ=0.1, 0.3, 0.5)
# ============================================================================

def plot_error_and_scatter_combined(df_010, df_030, df_050, save_path=None):
    """
    Create a combined plot with KDE comparison and scatter plot for ζ=0.1, 0.3, and 0.5.
    
    Parameters:
    -----------
    df_010 : pd.DataFrame
        Dataframe with zeta=0.1 uncertainty
    df_030 : pd.DataFrame
        Dataframe with zeta=0.3 uncertainty
    df_050 : pd.DataFrame
        Dataframe with zeta=0.5 uncertainty
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========== Subplot a) KDE comparison ==========
    ax1 = axes[0]
    
    # Plot KDE for each zeta value
    df_010['prediction_error_pct'].plot(kind='kde', ax=ax1, label='ζ=0.10', 
                                        color='blue', linewidth=2.5, alpha=0.8)
    df_030['prediction_error_pct'].plot(kind='kde', ax=ax1, label='ζ=0.30', 
                                        color='orange', linewidth=2.5, alpha=0.8)
    df_050['prediction_error_pct'].plot(kind='kde', ax=ax1, label='ζ=0.50', 
                                        color='red', linewidth=2.5, alpha=0.8)
    
    # Add zero error reference line
    ax1.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Zero Error')
    
    # Labels and formatting
    ax1.set_xlabel('Prediction Error (%)\n(a)', fontsize=13)
    ax1.set_ylabel('Density', fontsize=13)
    ax1.set_title('Comparison of Error Distributions', fontsize=14)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=11)
    ax1.set_xlim(-150, 150)
    
    # ========== Subplot b) Scatter plot for zeta=0.5 ========== 
    ax2 = axes[1]
    
    # Sample for visualization (plotting all 100k points is slow)
    df_sample = df_050.sample(n=min(5000, len(df_050)), random_state=42)
    
    ax2.scatter(df_sample['duration_actual'] / 60, 
               df_sample['duration_predicted'] / 60,
               alpha=0.3, s=10, color='steelblue', edgecolor='none')
    
    # Perfect prediction line
    max_dur = max(df_sample['duration_actual'].max(), df_sample['duration_predicted'].max()) / 60
    ax2.plot([0, max_dur], [0, max_dur], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Duration (minutes)\n(b)', fontsize=13)
    ax2.set_ylabel('Predicted Duration (minutes)', fontsize=13)
    ax2.set_title('Actual vs Predicted Task Duration ($\zeta$=0.50)', 
                 fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=11)
    ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Create the combined plot
plot_error_and_scatter_combined(
    df_zeta_010,
    df_zeta_030,
    df_zeta_050,
    save_path='../PLOTS/error_and_scatter_combined_zeta_010_030_050.png'
)



# 
# %% COMBINED PLOT: Error Distributions + Scatter Plot (ζ=0.1, 0.3, 0.5) [LOG-LOG SCATTER]

def plot_error_and_scatter_combined_loglog(df_010, df_030, df_050, save_path=None):
    """
    Create a combined plot with KDE comparison and scatter plot (log-log scale) for ζ=0.1, 0.3, and 0.5.
    
    Parameters:
    -----------
    df_010 : pd.DataFrame
        Dataframe with zeta=0.1 uncertainty
    df_030 : pd.DataFrame
        Dataframe with zeta=0.3 uncertainty
    df_050 : pd.DataFrame
        Dataframe with zeta=0.5 uncertainty
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # ========== Subplot a) KDE comparison ========== 
    ax1 = axes[0]
    # Plot KDE for each zeta value
    df_010['prediction_error_pct'].plot(kind='kde', ax=ax1, label='ζ=0.10', 
                                        color='blue', linewidth=2.5, alpha=0.8)
    df_030['prediction_error_pct'].plot(kind='kde', ax=ax1, label='ζ=0.30', 
                                        color='orange', linewidth=2.5, alpha=0.8)
    df_050['prediction_error_pct'].plot(kind='kde', ax=ax1, label='ζ=0.50', 
                                        color='red', linewidth=2.5, alpha=0.8)
    ax1.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Zero Error')
    ax1.set_xlabel('Prediction Error (%)\n(a)', fontsize=13)
    ax1.set_ylabel('Density (-)', fontsize=13)
    ax1.set_title('Comparison of Error Distributions', fontsize=14)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=11)
    ax1.set_xlim(-150, 150)
    # ========== Subplot b) Scatter plot for zeta=0.5 (log-log) ========== 
    ax2 = axes[1]
    df_sample = df_050.sample(n=min(5000, len(df_050)), random_state=42)
    x = df_sample['duration_actual'] / 60
    y = df_sample['duration_predicted'] / 60
    ax2.scatter(x, y,
               alpha=0.3, s=10, color='steelblue', edgecolor='none')
    max_dur = max(x.max(), y.max())
    ax2.plot([0.1, max_dur], [0.1, max_dur], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Duration (minutes)\n(b)', fontsize=13)
    ax2.set_ylabel('Predicted Duration (minutes)', fontsize=13)
    ax2.set_title('Actual vs Predicted Task Duration ($\\zeta$=0.50)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(0.1, max_dur)
    ax2.set_ylim(0.1, max_dur)
    ax2.grid(alpha=0.15, which='major')
    ax2.grid(False, which='minor')
    ax2.tick_params(labelsize=11)
    ax2.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Create the combined log-log plot
plot_error_and_scatter_combined_loglog(
    df_zeta_010,
    df_zeta_030,
    df_zeta_050,
    save_path='../PLOTS/error_and_scatter_combined_zeta_010_030_050_loglog.png'
)
# %%
