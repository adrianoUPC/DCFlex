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

# Example 5: Higher uncertainty (50% error)
df_gamma_050 = apply_duration_uncertainty(df_tasks, gamma=0.50, random_seed=42)

# Example 6: Very high uncertainty (70% error)
df_gamma_070 = apply_duration_uncertainty(df_tasks, gamma=0.70, random_seed=42)

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
print_uncertainty_stats(df_gamma_050, "Higher (γ=0.50)")
print_uncertainty_stats(df_gamma_070, "Very High (γ=0.70)")

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

# ============================================================================
# STANDALONE PLOT: Comparison of Error Distributions (ALL GAMMAS)
# ============================================================================

def plot_all_gamma_error_distributions(dataframes_dict, save_path=None):
    """
    Create a comprehensive comparison of error distributions across all gamma values.
    
    Parameters:
    -----------
    dataframes_dict : dict
        Dictionary mapping gamma values to their corresponding dataframes
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Color palette for different gamma values
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A040F', '#370617']
    gamma_values = list(dataframes_dict.keys())
    
    # Plot 1: KDE of Percentage Errors
    ax1 = axes[0, 0]
    for (gamma, df), color in zip(dataframes_dict.items(), colors):
        if "prediction_error_pct" in df.columns:
            df['prediction_error_pct'].plot(kind='kde', ax=ax1, label=f'γ={gamma}', 
                                           color=color, linewidth=2.5, alpha=0.8)
    ax1.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Zero Error')
    ax1.set_xlabel('Prediction Error (%)', fontsize=13)
    ax1.set_ylabel('Density', fontsize=13)
    ax1.set_title('Error Distribution Comparison (KDE)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=11)
    
    # Plot 2: Box Plot of Absolute Percentage Errors
    ax2 = axes[0, 1]
    box_data = []
    box_labels = []
    box_colors = []
    for idx, (gamma, df) in enumerate(dataframes_dict.items()):
        if "prediction_error_pct" in df.columns:
            box_data.append(df['prediction_error_pct'].abs())
            box_labels.append(f'γ={gamma}')
            box_colors.append(colors[idx])
    
    bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True, 
                     notch=True, showmeans=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Absolute Error (%)', fontsize=13)
    ax2.set_title('Error Magnitude Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(labelsize=11)
    
    # Plot 3: Cumulative Distribution Function
    ax3 = axes[1, 0]
    for (gamma, df), color in zip(dataframes_dict.items(), colors):
        if "prediction_error_pct" in df.columns:
            sorted_errors = np.sort(df['prediction_error_pct'].abs())
            cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax3.plot(sorted_errors, cumulative, label=f'γ={gamma}', 
                    color=color, linewidth=2.5, alpha=0.8)
    ax3.set_xlabel('Absolute Prediction Error (%)', fontsize=13)
    ax3.set_ylabel('Cumulative Probability', fontsize=13)
    ax3.set_title('Cumulative Distribution of Errors', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3)
    ax3.tick_params(labelsize=11)
    ax3.set_xlim(0, 100)
    
    # Plot 4: Summary Statistics Table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Compute statistics
    stats_data = []
    for gamma, df in dataframes_dict.items():
        if "prediction_error_pct" in df.columns:
            stats_data.append([
                f'γ={gamma}',
                f"{df['prediction_error'].abs().mean():.1f}",
                f"{df['prediction_error_pct'].abs().mean():.1f}%",
                f"{df['prediction_error_pct'].std():.1f}%",
                f"{df['prediction_error_pct'].abs().quantile(0.50):.1f}%",
                f"{df['prediction_error_pct'].abs().quantile(0.95):.1f}%"
            ])
    
    table = ax4.table(cellText=stats_data,
                     colLabels=['Gamma', 'MAE (s)', 'MAPE', 'Std Dev', 'Median', 'P95'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats_data) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    ax4.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Comprehensive Error Distribution Analysis Across All Gamma Values', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Create comprehensive comparison plot
gamma_dataframes = {
    0.10: df_gamma_010,
    0.20: df_gamma_020,
    0.30: df_gamma_030,
    0.50: df_gamma_050,
    0.70: df_gamma_070
}

plot_all_gamma_error_distributions(
    gamma_dataframes, 
    save_path='../PLOTS/all_gamma_error_comparison.png'
)

# ============================================================================
# SAMPLE SIZE SENSITIVITY ANALYSIS
# ============================================================================

def analyze_sample_size_impact(df_base, gamma=0.20, sample_sizes=[100, 500, 1000, 5000, 10000, 50000, 100000], random_seed=42):
    """
    Analyze how error statistics change with different sample sizes.
    
    Parameters:
    -----------
    df_base : pd.DataFrame
        Base dataframe with tasks
    gamma : float
        Uncertainty level to test
    sample_sizes : list
        List of sample sizes to test
    random_seed : int
        Random seed for reproducibility
    """
    results = []
    
    for n in sample_sizes:
        if n > len(df_base):
            continue
        
        # Sample the data
        df_sample = df_base.sample(n=n, random_state=random_seed)
        
        # Apply uncertainty
        df_uncertain = apply_duration_uncertainty(df_sample, gamma=gamma, random_seed=random_seed)
        
        # Calculate statistics
        mae = df_uncertain['prediction_error'].abs().mean()
        mape = df_uncertain['prediction_error_pct'].abs().mean()
        std_dev = df_uncertain['prediction_error_pct'].std()
        
        results.append({
            'sample_size': n,
            'mae': mae,
            'mape': mape,
            'std_dev': std_dev
        })
    
    df_results = pd.DataFrame(results)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Mean Absolute Error
    ax1 = axes[0]
    ax1.plot(df_results['sample_size'], df_results['mae'], 
            marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Sample Size (log scale)', fontsize=13)
    ax1.set_ylabel('Mean Absolute Error (seconds)', fontsize=13)
    ax1.set_title(f'MAE Convergence (γ={gamma})', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=11)
    
    # Plot 2: Mean Absolute Percentage Error
    ax2 = axes[1]
    ax2.plot(df_results['sample_size'], df_results['mape'], 
            marker='s', linewidth=2.5, markersize=8, color='#F18F01')
    ax2.set_xlabel('Sample Size (log scale)', fontsize=13)
    ax2.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=13)
    ax2.set_title(f'MAPE Convergence (γ={gamma})', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=11)
    
    # Plot 3: Standard Deviation
    ax3 = axes[2]
    ax3.plot(df_results['sample_size'], df_results['std_dev'], 
            marker='^', linewidth=2.5, markersize=8, color='#C73E1D')
    ax3.set_xlabel('Sample Size (log scale)', fontsize=13)
    ax3.set_ylabel('Standard Deviation (%)', fontsize=13)
    ax3.set_title(f'Std Dev Convergence (γ={gamma})', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(alpha=0.3)
    ax3.tick_params(labelsize=11)
    
    plt.suptitle('Sample Size Sensitivity Analysis: Statistical Convergence', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('../PLOTS/sample_size_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_results

# Run sample size analysis
print("\n" + "="*70)
print("Running sample size sensitivity analysis...")
print("="*70)
sample_size_results = analyze_sample_size_impact(
    df_tasks, 
    gamma=0.20, 
    sample_sizes=[100, 500, 1000, 5000, 10000, 50000, 100000]
)
print("\nSample Size Analysis Results:")
print(sample_size_results)

# ============================================================================
# ERROR MAGNITUDE VS TASK DURATION
# ============================================================================

def plot_error_vs_duration(dataframes_dict, save_path=None):
    """
    Analyze how prediction error relates to task duration across gamma values.
    
    Parameters:
    -----------
    dataframes_dict : dict
        Dictionary mapping gamma values to their corresponding dataframes
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A040F']
    
    for idx, (gamma, df) in enumerate(dataframes_dict.items()):
        if "prediction_error" not in df.columns or idx >= 6:
            continue
        
        ax = axes[idx]
        
        # Sample for visualization
        df_sample = df.sample(n=min(10000, len(df)), random_state=42)
        
        # Scatter plot with hexbin for density
        hb = ax.hexbin(df_sample['duration_actual'] / 60, 
                      df_sample['prediction_error'].abs() / 60,
                      gridsize=30, cmap='YlOrRd', mincnt=1, alpha=0.8)
        
        ax.set_xlabel('Actual Task Duration (minutes)', fontsize=11)
        ax.set_ylabel('Absolute Error (minutes)', fontsize=11)
        ax.set_title(f'γ={gamma}: Error vs Duration', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Add colorbar
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label('Count', fontsize=10)
    
    # Hide unused subplots
    for idx in range(len(dataframes_dict), 6):
        axes[idx].axis('off')
    
    plt.suptitle('Prediction Error Magnitude vs Task Duration', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

plot_error_vs_duration(
    gamma_dataframes,
    save_path='../PLOTS/error_vs_duration_analysis.png'
)

# # Exporting
# df_gamma_010.to_csv("../DATA/df_gamma_010.csv", index=False)
# df_gamma_020.to_csv("../DATA/df_gamma_020.csv", index=False)
# df_gamma_030.to_csv("../DATA/df_gamma_030.csv", index=False)
# df_gamma_050.to_csv("../DATA/df_gamma_050.csv", index=False)
# df_gamma_070.to_csv("../DATA/df_gamma_070.csv", index=False)

# # Export choosing the columns so that it matches df_tasks structure
# # Reset index to include task_name as a column
# cols_to_keep = ["task_name", "start_time", "end_time", "wait_time", "runtime_i", "plan_power"]
# df_gamma_010.reset_index()[cols_to_keep].to_csv("../DATA/df_tasks_gamma_010.csv", index=False)
# df_gamma_020.reset_index()[cols_to_keep].to_csv("../DATA/df_tasks_gamma_020.csv", index=False)
# df_gamma_030.reset_index()[cols_to_keep].to_csv("../DATA/df_tasks_gamma_030.csv", index=False)
# df_gamma_050.reset_index()[cols_to_keep].to_csv("../DATA/df_tasks_gamma_050.csv", index=False)
# df_gamma_070.reset_index()[cols_to_keep].to_csv("../DATA/df_tasks_gamma_070.csv", index=False)

# ============================================================================
# STANDALONE PLOT: Comparison of Error Distributions (γ=0.1, 0.3, 0.5)
# ============================================================================

def plot_error_distributions_standalone(df_010, df_030, df_050, save_path=None):
    """
    Create a standalone KDE comparison plot for γ=0.1, 0.3, and 0.5.
    
    Parameters:
    -----------
    df_010 : pd.DataFrame
        Dataframe with gamma=0.1 uncertainty
    df_030 : pd.DataFrame
        Dataframe with gamma=0.3 uncertainty
    df_050 : pd.DataFrame
        Dataframe with gamma=0.5 uncertainty
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot KDE for each gamma value
    df_010['prediction_error_pct'].plot(kind='kde', ax=ax, label='γ=0.10', 
                                        color='blue', linewidth=2.5, alpha=0.8)
    df_030['prediction_error_pct'].plot(kind='kde', ax=ax, label='γ=0.30', 
                                        color='orange', linewidth=2.5, alpha=0.8)
    df_050['prediction_error_pct'].plot(kind='kde', ax=ax, label='γ=0.50', 
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
    df_gamma_010,
    df_gamma_030,
    df_gamma_050,
    save_path='../PLOTS/error_distributions_gamma_010_030_050.png'
)

# ============================================================================
# COMBINED PLOT: Error Distributions + Scatter Plot (γ=0.1, 0.3, 0.5)
# ============================================================================

def plot_error_and_scatter_combined(df_010, df_030, df_050, save_path=None):
    """
    Create a combined plot with KDE comparison and scatter plot for γ=0.1, 0.3, and 0.5.
    
    Parameters:
    -----------
    df_010 : pd.DataFrame
        Dataframe with gamma=0.1 uncertainty
    df_030 : pd.DataFrame
        Dataframe with gamma=0.3 uncertainty
    df_050 : pd.DataFrame
        Dataframe with gamma=0.5 uncertainty
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========== Subplot a) KDE comparison ==========
    ax1 = axes[0]
    
    # Plot KDE for each gamma value
    df_010['prediction_error_pct'].plot(kind='kde', ax=ax1, label='γ=0.10', 
                                        color='blue', linewidth=2.5, alpha=0.8)
    df_030['prediction_error_pct'].plot(kind='kde', ax=ax1, label='γ=0.30', 
                                        color='orange', linewidth=2.5, alpha=0.8)
    df_050['prediction_error_pct'].plot(kind='kde', ax=ax1, label='γ=0.50', 
                                        color='red', linewidth=2.5, alpha=0.8)
    
    # Add zero error reference line
    ax1.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Zero Error')
    
    # Labels and formatting
    ax1.set_xlabel('Prediction Error (%)\na)', fontsize=13)
    ax1.set_ylabel('Density', fontsize=13)
    ax1.set_title('Comparison of Error Distributions', fontsize=14)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=11)
    ax1.set_xlim(-150, 150)
    
    # ========== Subplot b) Scatter plot for gamma=0.5 ==========
    ax2 = axes[1]
    
    # Sample for visualization (plotting all 100k points is slow)
    df_sample = df_050.sample(n=min(5000, len(df_050)), random_state=42)
    
    ax2.scatter(df_sample['duration_actual'] / 60, 
               df_sample['duration_predicted'] / 60,
               alpha=0.3, s=10, color='steelblue', edgecolor='none')
    
    # Perfect prediction line
    max_dur = max(df_sample['duration_actual'].max(), df_sample['duration_predicted'].max()) / 60
    ax2.plot([0, max_dur], [0, max_dur], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Duration (minutes)\nb)', fontsize=13)
    ax2.set_ylabel('Predicted Duration (minutes)', fontsize=13)
    ax2.set_title('Actual vs Predicted Task Duration (γ=0.50)', 
                 fontsize=14, fontweight='bold')
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
    df_gamma_010,
    df_gamma_030,
    df_gamma_050,
    save_path='../PLOTS/error_and_scatter_combined_gamma_010_030_050.png'
)
