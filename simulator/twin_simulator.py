import numpy as np

class DigitalTwinSimulator:
    def __init__(self):
        pass

    def simulate(
        self,
        age=22,
        weight=65.0,
        height=170.0,
        gender="Male",
        activity="Moderate",
        steps=8000,
        heart_rate=75,
        sleep_hours=7.5,
        active_calories=400,
        goal="Weight Loss",
        sim_steps=12000,
        sim_sleep=8.5,
        sim_active_cal=600,
        sim_heart_rate=68
    ):
        # 1. Base BMR
        if gender == "Male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        activity_factors = {
            "Sedentary": 1.20,
            "Light": 1.375,
            "Moderate": 1.55,
            "Very Active": 1.725,
            "Athlete": 1.90
        }
        factor = activity_factors.get(activity, 1.55)
        base_tdee = bmr * factor

        # 2. Current Digital Twin Score
        base_score = self._calc_score(steps, heart_rate, sleep_hours, active_calories)
        
        # 3. Simulated Digital Twin Score
        sim_score = self._calc_score(sim_steps, sim_heart_rate, sim_sleep, sim_active_cal)

        # 4. Metabolic Shift
        step_delta = sim_steps - steps
        cal_burn_delta = (step_delta * 0.04) + (sim_active_cal - active_calories)
        sim_tdee = base_tdee + cal_burn_delta

        # 5. Metabolic Age
        # Standard biological marker adjustments
        metabolic_age_offset = 0
        if sim_score >= 85:
            metabolic_age_offset = -3
        elif sim_score >= 70:
            metabolic_age_offset = -1
        elif sim_score < 50:
            metabolic_age_offset = +4

        sim_metabolic_age = max(18, age + metabolic_age_offset)

        # 6. Recovery & Glycemic Strain Index
        recovery_index = min(100, int((sim_sleep / 8.0) * 50 + (1 - max(0, sim_heart_rate - 60)/60) * 50))
        strain_index = min(100, int((sim_steps / 15000) * 60 + (sim_active_cal / 800) * 40))

        # 7. Macro recommendations adaptation
        if goal == "Weight Loss":
            rec_cal = sim_tdee - 450
            rec_protein = weight * 2.0
        elif goal == "Weight Gain":
            rec_cal = sim_tdee + 350
            rec_protein = weight * 1.8
        else:
            rec_cal = sim_tdee
            rec_protein = weight * 1.6

        rec_fat = (rec_cal * 0.25) / 9.0
        rec_carbs = (rec_cal - (rec_protein * 4 + rec_fat * 9)) / 4.0

        insights = []
        if sim_score > base_score + 10:
            insights.append(f"🔥 **+{(sim_score - base_score):.1f} pts** increase in Digital Twin health score through enhanced recovery and NEAT activity.")
        if sim_sleep >= 8.0 and sleep_hours < 7.0:
            insights.append("😴 Optimal sleep duration stimulates leptin signaling and reduces cortisol-mediated metabolic drag.")
        if sim_steps >= 10000:
            insights.append("👣 Hitting 10k+ steps improves postprandial glucose disposal and whole-body insulin sensitivity.")

        return {
            "base_score": round(base_score, 1),
            "sim_score": round(sim_score, 1),
            "base_tdee": round(base_tdee, 1),
            "sim_tdee": round(sim_tdee, 1),
            "tdee_delta": round(cal_burn_delta, 1),
            "base_metabolic_age": age,
            "sim_metabolic_age": sim_metabolic_age,
            "recovery_index": recovery_index,
            "strain_index": strain_index,
            "recommended_calories": round(rec_cal, 1),
            "recommended_protein": round(rec_protein, 1),
            "recommended_carbs": round(rec_carbs, 1),
            "recommended_fat": round(rec_fat, 1),
            "insights": insights
        }

    def _calc_score(self, steps, hr, sleep, active_cal):
        act_score = 100 if steps >= 10000 else (85 if steps >= 8000 else (70 if steps >= 6000 else 45))
        slp_score = 100 if sleep >= 8.0 else (85 if sleep >= 7.0 else (65 if sleep >= 6.0 else 35))
        hr_score = 100 if 55 <= hr <= 75 else (80 if 75 < hr <= 85 else 50)
        cal_score = 100 if active_cal >= 500 else (80 if active_cal >= 350 else 50)
        return act_score * 0.30 + slp_score * 0.25 + hr_score * 0.20 + cal_score * 0.25
