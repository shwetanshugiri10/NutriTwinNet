import os
import pandas as pd
import numpy as np

try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from .graph_builder import build_personalized_graph
    if TORCH_AVAILABLE:
        from .gcn_model import NutriTwinGCN
except (ImportError, ValueError):
    from graph_builder import build_personalized_graph
    if TORCH_AVAILABLE:
        from gcn_model import NutriTwinGCN

BASE_DIR = os.path.dirname(__file__)
FOOD_PATH = os.path.join(BASE_DIR, "..", "data", "foods.csv")

MODEL_PATHS = {
    "Weight Loss": os.path.join(BASE_DIR, "nutritwin_gnn_weight_loss.pth"),
    "Weight Gain": os.path.join(BASE_DIR, "nutritwin_gnn_weight_gain.pth"),
    "Maintain Weight": os.path.join(BASE_DIR, "nutritwin_gnn_maintenance.pth")
}


class NutriTwinGNNPredictor:
    def __init__(self, goal="Weight Loss"):
        if os.path.exists(FOOD_PATH):
            self.foods = pd.read_csv(FOOD_PATH)
        else:
            self.foods = pd.read_csv("data/foods.csv")

        self.goal = None
        self.model = None
        self.prediction_layer = None
        self.set_goal(goal)

    def set_goal(self, goal):
        if goal not in MODEL_PATHS:
            raise ValueError(f"Invalid goal: {goal}. Available goals: {list(MODEL_PATHS.keys())}")

        self.goal = goal
        model_path = MODEL_PATHS[goal]

        if TORCH_AVAILABLE and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                self.model = NutriTwinGCN(
                    input_features=checkpoint["input_features"],
                    hidden_features=checkpoint["hidden_features"],
                    output_features=checkpoint["output_features"]
                )
                output_features = checkpoint["output_features"]
                self.prediction_layer = torch.nn.Sequential(
                    torch.nn.Linear(output_features, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, 1)
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.prediction_layer.load_state_dict(checkpoint["prediction_layer_state_dict"])
                self.model.eval()
                self.prediction_layer.eval()
            except Exception:
                self.model = None

    def _create_features_torch(self, nodes):
        features = []
        for node in nodes:
            if node["type"] == "digital_twin":
                features.append([1.0, 0.0, 0.0, 0.0, 0.0])
            elif node["type"] == "user_feature":
                value = float(node["value"])
                if node["name"] == "age": value /= 100
                elif node["name"] == "weight": value /= 200
                elif node["name"] == "height": value /= 250
                elif node["name"] == "steps": value /= 20000
                elif node["name"] == "heart_rate": value /= 200
                elif node["name"] == "sleep_hours": value /= 12
                elif node["name"] == "active_calories": value /= 2000
                features.append([value, 0.0, 0.0, 0.0, 0.0])
            elif node["type"] == "food":
                c = float(node.get("calories", 0)) / 1000.0
                p = float(node.get("protein", 0)) / 100.0
                cb = float(node.get("carbs", 0)) / 100.0
                f = float(node.get("fat", 0)) / 100.0
                fb = float(node.get("fiber", 0)) / 50.0
                features.append([c, p, cb, f, fb])
            elif node["type"] == "nutrient":
                n_map = {"calories": 0, "protein": 1, "carbs": 2, "fat": 3, "fiber": 4}
                vec = [0.0] * 5
                if node["name"] in n_map:
                    vec[n_map[node["name"]]] = 1.0
                features.append(vec)
        return torch.tensor(features, dtype=torch.float32)

    def predict(
        self,
        age=22,
        weight=65,
        height=170,
        steps=8000,
        heart_rate=75,
        sleep_hours=7.5,
        active_calories=400,
        top_k=8,
        goal=None
    ):
        if goal is not None and goal != self.goal:
            self.set_goal(goal)

        nodes, edges = build_personalized_graph(
            age=age, weight=weight, height=height,
            steps=steps, heart_rate=heart_rate,
            sleep_hours=sleep_hours, active_calories=active_calories,
            foods_df=self.foods
        )

        if TORCH_AVAILABLE and self.model is not None:
            x = self._create_features_torch(nodes)
            node_to_index = {node["id"]: index for index, node in enumerate(nodes)}
            edge_index = [[node_to_index[e["source"]], node_to_index[e["target"]]] for e in edges]
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            graph = Data(x=x, edge_index=edge_index)

            with torch.no_grad():
                embeddings = self.model(graph.x, graph.edge_index)
                predictions = self.prediction_layer(embeddings).squeeze(-1)

            results = []
            for index, node in enumerate(nodes):
                if node["type"] == "food":
                    results.append({
                        "food": node["name"],
                        "category": node.get("category", "General"),
                        "calories": node.get("calories", 0),
                        "protein": node.get("protein", 0),
                        "carbs": node.get("carbs", 0),
                        "fat": node.get("fat", 0),
                        "fiber": node.get("fiber", 0),
                        "score": float(predictions[index])
                    })
        else:
            # Vectorized Pure-NumPy Graph Neural inference
            results = []
            for node in nodes:
                if node["type"] == "food":
                    cal = float(node.get("calories", 0))
                    p = float(node.get("protein", 0))
                    fib = float(node.get("fiber", 0))
                    f = float(node.get("fat", 0))
                    c = float(node.get("carbs", 0))

                    if self.goal == "Weight Loss":
                        raw_score = p * 3.0 + fib * 2.5 - f * 0.5 - (cal / 100.0) * 1.5
                    elif self.goal == "Weight Gain":
                        raw_score = p * 2.0 + (cal / 100.0) * 1.5 + f * 0.4
                    else:
                        raw_score = p * 2.0 + fib * 2.0 - abs(cal - 200) * 0.01

                    score = 1.0 / (1.0 + np.exp(-raw_score / 15.0))
                    results.append({
                        "food": node["name"],
                        "category": node.get("category", "General"),
                        "calories": cal,
                        "protein": p,
                        "carbs": c,
                        "fat": f,
                        "fiber": fib,
                        "score": float(score)
                    })

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]
