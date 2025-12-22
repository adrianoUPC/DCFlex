import pandas as pd

def return_params(simulation, scenario):
    if simulation == "DEFAULT":
        if scenario == "BASELINE":
            latencies = [600]
            flex_windows = [pd.Timedelta(hours=3)]        
            delta_notifications = [pd.Timedelta(minutes=60)]  
            betas = [0.7]                                  
            gamma_buffers = [pd.Timedelta(minutes=60)]    

        elif scenario == "DAM":
            latencies = [60*30]
            flex_windows = [pd.Timedelta(hours=4)]        
            delta_notifications = [pd.Timedelta(minutes=60*12)]  
            betas = [0.8]                                  
            gamma_buffers = [pd.Timedelta(minutes=120)]
            
    elif simulation == "SENSITIVE":
        if scenario == "BASELINE":
            latencies = [600, 900]  # seconds
            flex_window_values = [1, 2]  # hours
            flex_windows = [pd.Timedelta(hours=h) for h in flex_window_values]
            delta_notifications = [pd.Timedelta(minutes=60)]  # fixed at 60 min
            betas = [0.5, 0.7]  # Overhang tolerance - SENSITIVITY ANALYSIS
            gamma_buffers = [pd.Timedelta(minutes=60)]
        elif scenario == "DAM":
            latencies = [60*15, 60*30]  # seconds
            flex_window_values = [4, 6]  # hours
            flex_windows = [pd.Timedelta(hours=h) for h in flex_window_values]
            delta_notifications = [pd.Timedelta(minutes=60*12), pd.Timedelta(minutes=60*24)]  
            betas = [0.7]  # Overhang tolerance - SENSITIVITY ANALYSIS
            gamma_buffers = [pd.Timedelta(minutes=120)]
            
    params = {
        'latencies': latencies,
        'flex_windows': flex_windows,
        'delta_notifications': delta_notifications,
        'betas': betas,
        'gamma_buffers': gamma_buffers
    }
    return params