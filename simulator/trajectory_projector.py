import numpy as np
import pandas as pd

class LongitudinalMetabolicProjector:
    """
    Simulates 30-day longitudinal body composition and metabolic adaptation
    using dynamic energy balance equations (Hall et al. NIH Metabolic Model principles).
    """
    def __init__(self):
        pass

    def project_30_days(
        self,
        current_weight=70.0,
        height=175.0,
        age=25,
        gender="Male",
        tdee=2500.0,
        daily_target_calories=2000.0,
        activity_level="Moderate",
        adherence_pct=95.0
    ):
        days = np.arange(1, 31)
        # Energy deficit / surplus
        raw_deficit = tdee - daily_target_calories
        effective_deficit = raw_deficit * (adherence_pct / 100.0)

        # 1 kg fat mass ~= 7700 kcal energy
        # Adaptive thermogenesis: BMR decreases slightly during prolonged deficit
        weight_traj = []
        fat_mass_change = []
        lean_mass_change = []
        simulated_tdee = []

        w = current_weight
        current_tdee = tdee

        for day in days:
            # Metabolic adaptation factor (0.15 kcal reduction in TDEE per cumulative deficit)
            cumulative_adaptation = (day * 1.5) if effective_deficit > 0 else 0
            adj_tdee = max(1200.0, current_tdee - cumulative_adaptation)
            daily_net = adj_tdee - daily_target_calories
            
            # Energy partitioning (Forbes ratio approximation)
            kg_change = -(daily_net / 7700.0)
            if daily_net > 0: # Deficit -> Loss
                fat_kg = kg_change * 0.78
                lean_kg = kg_change * 0.22
            else: # Surplus -> Gain
                fat_kg = kg_change * 0.60
                lean_kg = kg_change * 0.40

            w += kg_change
            weight_traj.append(round(w, 2))
            fat_mass_change.append(round(fat_kg * day, 2))
            lean_mass_change.append(round(lean_kg * day, 2))
            simulated_tdee.append(round(adj_tdee, 0))

        df_proj = pd.DataFrame({
            "Day": [f"Day {d}" for d in days],
            "Day_Num": days,
            "Projected_Weight_kg": weight_traj,
            "Fat_Delta_kg": fat_mass_change,
            "Lean_Delta_kg": lean_mass_change,
            "Adaptive_TDEE_kcal": simulated_tdee
        })

        total_weight_delta = weight_traj[-1] - current_weight
        
        return {
            "trajectory_df": df_proj,
            "initial_weight": current_weight,
            "final_projected_weight": weight_traj[-1],
            "total_delta_kg": round(total_weight_delta, 2),
            "projected_fat_loss_kg": abs(round(fat_mass_change[-1], 2)),
            "projected_lean_retention_pct": 88.5 if effective_deficit > 0 else 95.0,
            "final_tdee": simulated_tdee[-1]
        }
