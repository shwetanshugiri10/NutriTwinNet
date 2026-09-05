import math


class NutriTwinWearableAnalyzer:

    def __init__(self):
        self.feature_names = [
            "steps",
            "heart_rate",
            "sleep_hours",
            "active_calories"
        ]

    # ========================================================
    # STEP ANALYSIS
    # ========================================================

    def analyze_steps(self, steps):

        steps = float(steps)

        if steps < 5000:
            status = "Low"
            score = 0.30

        elif steps < 8000:
            status = "Moderate"
            score = 0.60

        elif steps < 12000:
            status = "Good"
            score = 0.85

        else:
            status = "High"
            score = 1.00

        return {
            "value": steps,
            "status": status,
            "score": score
        }

    # ========================================================
    # HEART RATE ANALYSIS
    # ========================================================

    def analyze_heart_rate(self, heart_rate):

        heart_rate = float(heart_rate)

        if heart_rate < 60:
            status = "Low"

        elif heart_rate <= 100:
            status = "Normal"

        else:
            status = "High"

        # Distance from a simple reference point.
        #
        # This is an educational prototype metric,
        # not a medical assessment.

        distance = abs(
            heart_rate - 75
        )

        score = max(
            0.0,
            1.0 - (distance / 50.0)
        )

        return {
            "value": heart_rate,
            "status": status,
            "score": score
        }

    # ========================================================
    # SLEEP ANALYSIS
    # ========================================================

    def analyze_sleep(self, sleep_hours):

        sleep_hours = float(sleep_hours)

        if sleep_hours < 6:
            status = "Low"
            score = 0.30

        elif sleep_hours < 7:
            status = "Below Optimal"
            score = 0.60

        elif sleep_hours <= 9:
            status = "Good"
            score = 1.00

        else:
            status = "High"
            score = 0.75

        return {
            "value": sleep_hours,
            "status": status,
            "score": score
        }

    # ========================================================
    # ACTIVE CALORIES ANALYSIS
    # ========================================================

    def analyze_active_calories(
        self,
        active_calories
    ):

        active_calories = float(
            active_calories
        )

        if active_calories < 200:
            status = "Low"
            score = 0.30

        elif active_calories < 400:
            status = "Moderate"
            score = 0.60

        elif active_calories < 700:
            status = "Good"
            score = 0.85

        else:
            status = "High"
            score = 1.00

        return {
            "value": active_calories,
            "status": status,
            "score": score
        }

    # ========================================================
    # OVERALL WEARABLE SCORE
    # ========================================================

    def calculate_overall_score(
        self,
        steps,
        heart_rate,
        sleep_hours,
        active_calories
    ):

        step_result = self.analyze_steps(
            steps
        )

        heart_result = self.analyze_heart_rate(
            heart_rate
        )

        sleep_result = self.analyze_sleep(
            sleep_hours
        )

        calorie_result = (
            self.analyze_active_calories(
                active_calories
            )
        )

        scores = [
            step_result["score"],
            heart_result["score"],
            sleep_result["score"],
            calorie_result["score"]
        ]

        overall_score = (
            sum(scores) / len(scores)
        )

        return round(
            overall_score,
            4
        )

    # ========================================================
    # COMPLETE ANALYSIS
    # ========================================================

    def analyze(
        self,
        steps,
        heart_rate,
        sleep_hours,
        active_calories
    ):

        step_result = self.analyze_steps(
            steps
        )

        heart_result = self.analyze_heart_rate(
            heart_rate
        )

        sleep_result = self.analyze_sleep(
            sleep_hours
        )

        calorie_result = (
            self.analyze_active_calories(
                active_calories
            )
        )

        overall_score = (
            self.calculate_overall_score(
                steps=steps,
                heart_rate=heart_rate,
                sleep_hours=sleep_hours,
                active_calories=active_calories
            )
        )

        return {
            "steps": step_result,
            "heart_rate": heart_result,
            "sleep_hours": sleep_result,
            "active_calories": calorie_result,
            "overall_score": overall_score
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NutriTwinNet Wearable Analytics")
    print("=" * 60)

    analyzer = NutriTwinWearableAnalyzer()

    result = analyzer.analyze(
        steps=10500,
        heart_rate=76,
        sleep_hours=7.5,
        active_calories=550
    )

    print()
    print("WEARABLE ANALYSIS")
    print("-" * 60)

    print(
        f"Steps           : "
        f"{result['steps']['value']:.0f} "
        f"→ {result['steps']['status']} "
        f"(score: {result['steps']['score']:.2f})"
    )

    print(
        f"Heart Rate      : "
        f"{result['heart_rate']['value']:.0f} bpm "
        f"→ {result['heart_rate']['status']} "
        f"(score: {result['heart_rate']['score']:.2f})"
    )

    print(
        f"Sleep           : "
        f"{result['sleep_hours']['value']:.1f} hours "
        f"→ {result['sleep_hours']['status']} "
        f"(score: {result['sleep_hours']['score']:.2f})"
    )

    print(
        f"Active Calories : "
        f"{result['active_calories']['value']:.0f} kcal "
        f"→ {result['active_calories']['status']} "
        f"(score: {result['active_calories']['score']:.2f})"
    )

    print()
    print(
        f"Overall Wearable Score: "
        f"{result['overall_score']:.4f}"
    )

    print()
    print("=" * 60)
    print(
        "Wearable analytics completed successfully."
    )
    print("=" * 60)