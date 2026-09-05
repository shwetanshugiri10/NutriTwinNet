import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
FOOD_PATH = os.path.join(BASE_DIR, "..", "data", "foods.csv")


def load_foods():
    if os.path.exists(FOOD_PATH):
        return pd.read_csv(FOOD_PATH)
    return pd.read_csv("data/foods.csv")


def build_personalized_graph(
    age=22,
    weight=65,
    height=170,
    steps=8000,
    heart_rate=75,
    sleep_hours=7.5,
    active_calories=400,
    foods_df=None
):
    if foods_df is None:
        foods = load_foods()
    else:
        foods = foods_df

    nodes = []
    edges = []

    # ==================================================
    # DIGITAL TWIN NODE
    # ==================================================

    user_node = {
        "id": "user_0",
        "type": "digital_twin",
        "name": "User Digital Twin"
    }

    nodes.append(user_node)

    # ==================================================
    # USER FEATURE NODES
    # ==================================================

    user_features = {
        "age": age,
        "weight": weight,
        "height": height,
        "steps": steps,
        "heart_rate": heart_rate,
        "sleep_hours": sleep_hours,
        "active_calories": active_calories
    }

    for feature, value in user_features.items():

        feature_node = {
            "id": f"user_feature_{feature}",
            "type": "user_feature",
            "name": feature,
            "value": value
        }

        nodes.append(feature_node)

        edges.append({
            "source": "user_0",
            "target": f"user_feature_{feature}",
            "weight": float(value)
        })

    # ==================================================
    # NUTRIENT NODES
    # ==================================================

    nutrients = [
        "calories",
        "protein",
        "carbs",
        "fat",
        "fiber"
    ]

    for nutrient in nutrients:

        nutrient_node = {
            "id": f"nutrient_{nutrient}",
            "type": "nutrient",
            "name": nutrient
        }

        nodes.append(nutrient_node)

    # ==================================================
    # FOOD NODES & EDGES
    # ==================================================

    for index, row in foods.iterrows():
        food_id = f"food_{index}"
        food_node = {
            "id": food_id,
            "type": "food",
            "name": row["food"],
            "category": row.get("category", "General"),
            "calories": float(row.get("calories", 0)),
            "protein": float(row.get("protein", 0)),
            "carbs": float(row.get("carbs", 0)),
            "fat": float(row.get("fat", 0)),
            "fiber": float(row.get("fiber", 0)),
            "food_index": index
        }
        nodes.append(food_node)

        # Connect Digital Twin to every food
        edges.append({
            "source": "user_0",
            "target": food_id,
            "weight": 1.0
        })

        # Food -> Nutrient edges
        for nutrient in nutrients:
            value = float(row.get(nutrient, 0))
            edges.append({
                "source": food_id,
                "target": f"nutrient_{nutrient}",
                "weight": value
            })

    return nodes, edges


# ======================================================
# TEST THE PERSONALIZED GRAPH
# ======================================================

if __name__ == "__main__":

    nodes, edges = build_personalized_graph(
        age=22,
        weight=65,
        height=170,
        steps=10500,
        heart_rate=76,
        sleep_hours=7.5,
        active_calories=550
    )

    print("=" * 60)
    print("NutriTwinNet Personalized GNN Graph")
    print("=" * 60)

    print(f"Total Nodes: {len(nodes)}")
    print(f"Total Edges: {len(edges)}")

    print("\nDigital Twin:")

    for node in nodes:

        if node["type"] == "digital_twin":
            print(node)

    print("\nUser Features:")

    for node in nodes:

        if node["type"] == "user_feature":
            print(node)

    print("\nSample Food Nodes:")

    food_count = 0

    for node in nodes:

        if node["type"] == "food":

            print(node)

            food_count += 1

            if food_count == 5:
                break

    print("\nSample Edges:")

    for edge in edges[:15]:
        print(edge)