import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_drift(reference_data, current_data, threshold=0.05):
    drift_report = {}
    is_drift_detected = False

    for col in reference_data.columns:
        if col in current_data.columns:
            stat, p_value = ks_2samp(reference_data[col], current_data[col])
            drift = p_value < threshold
            drift_report[col] = {
                "p_value": float(p_value),
                "drift_detected": bool(drift)
            }
            if drift:
                is_drift_detected = True
    return is_drift_detected, drift_report

if __name__ == "__main__":
    print("Script shuru ho gayi hai...")
    # Dummy data
    ref_data = pd.DataFrame({'val': np.random.normal(0, 1, 100)})
    curr_data = pd.DataFrame({'val': np.random.normal(0.5, 1, 100)}) 
    
    drift, report = detect_drift(ref_data, curr_data)
    
    print(f"Drift Detected: {drift}")
    print(f"Report: {json.dumps(report, indent=4)}")