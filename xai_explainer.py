import pandas as pd


class NutriTwinXAI:

    def __init__(self, food_path="data/foods.csv"):

        self.foods = pd.read_csv(food_path)


    def explain_food(
        self,
        food_name,
        score,
        goal,
        age,
        weight,
        height,
        steps,
        heart_rate,
        sleep_hours,
        active_calories
    ):

        matches = self.foods[
            self.foods["food"] == food_name
        ]

        if matches.empty:
            return {
                "food": food_name,
                "score": score,
                "reasons": [
                    "Food information was not found."
                ]
            }

        food = matches.iloc[0]

        reasons = []

        calories = float(food["calories"])
        protein = float(food["protein"])
        carbs = float(food["carbs"])
        fat = float(food["fat"])
        fiber = float(food["fiber"])

        # --------------------------------------------------
        # Goal-specific explanations
        # --------------------------------------------------

        if goal == "Weight Gain":

            if calories >= 200:
                reasons.append(
                    f"High calorie density ({calories:.0f} kcal) "
                    "supports the weight-gain objective."
                )

            if protein >= 15:
                reasons.append(
                    f"High protein content ({protein:.1f} g) "
                    "supports muscle and tissue development."
                )

            if fat >= 10:
                reasons.append(
                    f"Provides substantial dietary fat "
                    f"({fat:.1f} g), increasing energy density."
                )

            if carbs >= 20:
                reasons.append(
                    f"Provides carbohydrates ({carbs:.1f} g) "
                    "for additional dietary energy."
                )

        elif goal == "Weight Loss":

            if protein >= 15:
                reasons.append(
                    f"High protein content ({protein:.1f} g) "
                    "can support satiety and protein intake."
                )

            if calories <= 250:
                reasons.append(
                    f"Moderate calorie content ({calories:.0f} kcal) "
                    "fits a calorie-conscious objective."
                )

            if fiber >= 3:
                reasons.append(
                    f"Contains fiber ({fiber:.1f} g), "
                    "which contributes to dietary satiety."
                )

        elif goal == "Maintain Weight":

            if protein >= 10:
                reasons.append(
                    f"Provides protein ({protein:.1f} g) "
                    "for a balanced dietary profile."
                )

            if 100 <= calories <= 300:
                reasons.append(
                    f"Moderate energy content ({calories:.0f} kcal) "
                    "fits a balanced maintenance profile."
                )

            if fiber >= 2:
                reasons.append(
                    f"Contains fiber ({fiber:.1f} g), "
                    "supporting nutritional balance."
                )

        # --------------------------------------------------
        # General nutritional explanation
        # --------------------------------------------------

        if not reasons:

            reasons.append(
                "The food received a favorable score from "
                "the trained GNN model for this objective."
            )

        # --------------------------------------------------
        # Personalized context
        # --------------------------------------------------

        context = []

        if steps >= 10000:
            context.append(
                f"High daily activity ({steps:,} steps)"
            )

        elif steps < 5000:
            context.append(
                f"Lower daily activity ({steps:,} steps)"
            )

        if sleep_hours >= 7:
            context.append(
                f"Sleep duration is {sleep_hours:.1f} hours"
            )

        if active_calories >= 500:
            context.append(
                f"Active calories are relatively high "
                f"({active_calories} kcal)"
            )

        # --------------------------------------------------
        # Explanation result
        # --------------------------------------------------

        return {

            "food": food_name,

            "score": float(score),

            "goal": goal,

            "nutrition": {

                "calories": calories,

                "protein": protein,

                "carbs": carbs,

                "fat": fat,

                "fiber": fiber

            },

            "reasons": reasons,

            "personalized_context": context

        }