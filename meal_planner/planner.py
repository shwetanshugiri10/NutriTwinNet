import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(__file__)
FOOD_PATH = os.path.join(BASE_DIR, "..", "data", "foods.csv")

def load_foods():
    if os.path.exists(FOOD_PATH):
        return pd.read_csv(FOOD_PATH)
    return pd.read_csv("data/foods.csv")

class NutriTwinMealPlanner:
    def __init__(self):
        self.foods = load_foods()

    def generate_daily_plan(
        self,
        target_calories=2000,
        target_protein=120,
        target_carbs=220,
        target_fat=65,
        goal="Weight Loss",
        diet_filter="All"
    ):
        df = self.foods.copy()

        # Filter out industrial powders and pure isolate powders unless high protein
        if diet_filter != "High Protein":
            unwanted_keywords = ["isolate", "dried powder", "vital wheat", "crude", "pure", "industrial"]
            for kw in unwanted_keywords:
                df = df[~df["food"].str.lower().str.contains(kw)]

        # Apply dietary filter
        if diet_filter == "Vegetarian":
            non_veg = ["Protein", "Poultry", "Fish & Seafood", "Beef", "Pork"]
            df = df[~df["category"].isin(non_veg)]
        elif diet_filter == "High Protein":
            df = df[df["protein"] >= 6]
        elif diet_filter == "Low Carb":
            df = df[df["carbs"] <= 25]
        elif diet_filter == "High Fiber":
            df = df[df["fiber"] >= 2.0]


        # Meal targets distribution
        splits = {
            "Breakfast": 0.25,
            "Lunch": 0.35,
            "Dinner": 0.30,
            "Snacks": 0.10
        }

        category_rules = {
            "Breakfast": ["Breakfast", "Grain", "Dairy", "Fruit"],
            "Lunch": ["Protein", "Grain", "Vegetable", "Legume", "Meals"],
            "Dinner": ["Protein", "Vegetable", "Legume", "Soups & Sauces", "Meals"],
            "Snacks": ["Fruit", "Nuts", "Snacks", "Dairy", "Beverage"]
        }

        daily_plan = {}
        total_cal, total_p, total_c, total_f, total_fib = 0, 0, 0, 0, 0
        total_sodium, total_potassium = 0, 0

        for meal_name, ratio in splits.items():
            meal_target_cal = target_calories * ratio
            meal_target_p = target_protein * ratio
            allowed_cats = category_rules[meal_name]

            meal_candidates = df[df["category"].isin(allowed_cats)].copy()
            if meal_candidates.empty:
                meal_candidates = df.copy()

            # Rank candidates based on macro fit
            meal_candidates["fit_score"] = (
                meal_candidates["protein"] * 2.0
                + meal_candidates["fiber"] * 1.5
                - abs(meal_candidates["calories"] - (meal_target_cal / 2.0)) * 0.05
            )

            # Pick 2 complementary items
            sorted_items = meal_candidates.sort_values(by="fit_score", ascending=False)
            
            # Select distinct items
            selected = []
            seen_cats = set()
            for _, item in sorted_items.iterrows():
                if item["category"] not in seen_cats or len(selected) < 1:
                    selected.append(item)
                    seen_cats.add(item["category"])
                if len(selected) == 2:
                    break

            if len(selected) < 2 and len(sorted_items) >= 2:
                selected = [sorted_items.iloc[0], sorted_items.iloc[1]]

            meal_items = []
            m_cal, m_p, m_c, m_f, m_fib = 0, 0, 0, 0, 0
            for item in selected:
                item_dict = {
                    "food": item["food"],
                    "category": item["category"],
                    "calories": float(item["calories"]),
                    "protein": float(item["protein"]),
                    "carbs": float(item["carbs"]),
                    "fat": float(item["fat"]),
                    "fiber": float(item["fiber"]),
                    "sodium": float(item.get("sodium", 0)),
                    "potassium": float(item.get("potassium", 0))
                }
                meal_items.append(item_dict)
                m_cal += item_dict["calories"]
                m_p += item_dict["protein"]
                m_c += item_dict["carbs"]
                m_f += item_dict["fat"]
                m_fib += item_dict["fiber"]
                total_sodium += item_dict["sodium"]
                total_potassium += item_dict["potassium"]

            total_cal += m_cal
            total_p += m_p
            total_c += m_c
            total_f += m_f
            total_fib += m_fib

            daily_plan[meal_name] = {
                "target_calories": round(meal_target_cal, 1),
                "actual_calories": round(m_cal, 1),
                "protein": round(m_p, 1),
                "carbs": round(m_c, 1),
                "fat": round(m_f, 1),
                "fiber": round(m_fib, 1),
                "items": meal_items
            }

        summary = {
            "target_calories": round(target_calories, 1),
            "total_calories": round(total_cal, 1),
            "total_protein": round(total_p, 1),
            "target_protein": round(target_protein, 1),
            "total_carbs": round(total_c, 1),
            "target_carbs": round(target_carbs, 1),
            "total_fat": round(total_f, 1),
            "target_fat": round(target_fat, 1),
            "total_fiber": round(total_fib, 1),
            "total_sodium": round(total_sodium, 1),
            "total_potassium": round(total_potassium, 1),
            "calorie_match_pct": round((total_cal / max(target_calories, 1)) * 100, 1),
            "meals": daily_plan
        }

        return summary
