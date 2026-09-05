import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)
FOOD_PATH = os.path.join(BASE_DIR, "..", "data", "foods.csv")


def load_food_data():
    if os.path.exists(FOOD_PATH):
        return pd.read_csv(FOOD_PATH)
    return pd.read_csv("data/foods.csv")


def recommend_foods(
    calories=2000,
    protein=100,
    goal="Weight Loss",
    category=None,
    search_query=None,
    diet_filter=None,
    limit=10
):
    foods = load_food_data().copy()

    # Category Filtering
    if category and category != "All":
        foods = foods[foods["category"] == category]

    # Keyword Search
    if search_query and search_query.strip():
        q = search_query.strip().lower()
        foods = foods[foods["food"].str.lower().str.contains(q, na=False)]

    # Dietary Filters
    if diet_filter == "High Protein":
        foods = foods[foods["protein"] >= 10]
    elif diet_filter == "Low Carb":
        foods = foods[foods["carbs"] <= 15]
    elif diet_filter == "High Fiber":
        foods = foods[foods["fiber"] >= 3]
    elif diet_filter == "Low Calorie":
        foods = foods[foods["calories"] <= 150]
    elif diet_filter == "Vegetarian":
        non_veg = ["Protein", "Poultry", "Fish & Seafood", "Beef", "Pork"]
        foods = foods[~foods["category"].isin(non_veg)]

    if foods.empty:
        return foods

    # Base Nutritional Quality Score
    foods["score"] = (
        foods["protein"] * 3.0
        + foods["fiber"] * 2.5
        - foods["fat"] * 0.4
    )

    # Goal Adjustments
    if goal == "Weight Loss":
        foods["score"] = (
            foods["score"]
            + foods["fiber"] * 2.0
            - (foods["calories"] / 100.0) * 1.5
        )
    elif goal == "Weight Gain":
        foods["score"] = (
            foods["score"]
            + (foods["calories"] / 100.0) * 1.2
            + foods["protein"] * 1.5
        )
    else:  # Maintain Weight / Balanced
        foods["score"] = (
            foods["score"]
            + foods["protein"] * 1.0
            + foods["fiber"] * 1.0
            - abs(foods["calories"] - 200) * 0.01
        )

    recommendations = foods.sort_values(
        by="score",
        ascending=False
    )

    return recommendations.head(limit)