import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class WearableTimeSeriesAnalyzer:
    def __init__(self):
        pass

    def get_7day_history(self, current_steps=8000, current_hr=75, current_sleep=7.5, current_active_cal=400):
        # Generate realistic 7-day wearable stream ending today
        today = datetime.now()
        dates = [(today - timedelta(days=6 - i)).strftime("%a (%b %d)") for i in range(7)]

        np.random.seed(42)
        # Synthetic realistic 7-day variance based on current baseline
        steps_hist = [int(current_steps * np.random.uniform(0.85, 1.25)) for _ in range(6)] + [current_steps]
        hr_hist = [int(current_hr * np.random.uniform(0.95, 1.08)) for _ in range(6)] + [current_hr]
        sleep_hist = [round(float(current_sleep + np.random.uniform(-0.8, 0.8)), 1) for _ in range(6)] + [current_sleep]
        active_cal_hist = [int(current_active_cal * np.random.uniform(0.85, 1.20)) for _ in range(6)] + [current_active_cal]
        hrv_hist = [int(65 + (s - 7.0)*8 - (h - 70)*0.5 + np.random.uniform(-4, 4)) for s, h in zip(sleep_hist, hr_hist)]

        df = pd.DataFrame({
            "Date": dates,
            "Steps": steps_hist,
            "Resting_HR": hr_hist,
            "Sleep_Hours": sleep_hist,
            "Active_Calories": active_cal_hist,
            "HRV_ms": hrv_hist
        })

        avg_steps = int(df["Steps"].mean())
        avg_sleep = round(df["Sleep_Hours"].mean(), 1)
        avg_hrv = int(df["HRV_ms"].mean())
        avg_hr = int(df["Resting_HR"].mean())

        anomalies = []
        if df["Sleep_Hours"].iloc[-1] < 6.0:
            anomalies.append("⚠️ **Sleep Deficit**: Last night's sleep was under 6h. Recommended recovery-focused nutrition.")
        if df["HRV_ms"].iloc[-1] < avg_hrv - 8:
            anomalies.append("⚠️ **Elevated Autonomic Strain**: HRV is 8ms below your 7-day baseline.")
        if df["Steps"].iloc[-1] >= avg_steps + 3000:
            anomalies.append("⚡ **High Caloric Expenditure Day**: Steps +3,000 above weekly average. Consider carb replenishment.")

        return {
            "data": df,
            "avg_steps": avg_steps,
            "avg_sleep": avg_sleep,
            "avg_hrv": avg_hrv,
            "avg_hr": avg_hr,
            "anomalies": anomalies
        }
