import pandas as pd
import numpy as np

    
def apply_duration_uncertainty(df, zeta, random_seed=42):
    """
    Apply normally distributed prediction error to task durations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe with 'runtime_i' and 'start_time'
    zeta : float
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
    
    # Generate prediction errors: ε_k ~ N(0, (ζ * d_k)^2)
    sigma = zeta * d_actual
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