import os

import pandas as pd
import torch
from torch_geometric.data import Data

from .graph_builder import build_personalized_graph
from .gcn_model import NutriTwinGCN


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(__file__)

FOOD_PATH = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "foods.csv"
)

MODEL_PATHS = {
    "Weight Loss": os.path.join(
        BASE_DIR,
        "nutritwin_gnn_weight_loss.pth"
    ),

    "Weight Gain": os.path.join(
        BASE_DIR,
        "nutritwin_gnn_weight_gain.pth"
    ),

    "Maintain Weight": os.path.join(
        BASE_DIR,
        "nutritwin_gnn_maintenance.pth"
    )
}


# ============================================================
# NUTRITWIN GNN PREDICTOR
# ============================================================

class NutriTwinGNNPredictor:

    def __init__(self, goal="Weight Loss"):

        self.foods = pd.read_csv(
            FOOD_PATH
        )

        self.goal = goal

        self._load_model(goal)


    # ========================================================
    # LOAD MODEL
    # ========================================================

    def _load_model(self, goal):

        if goal not in MODEL_PATHS:

            raise ValueError(
                f"Invalid goal: {goal}\n"
                f"Available goals: "
                f"{list(MODEL_PATHS.keys())}"
            )

        model_path = MODEL_PATHS[goal]

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"Model file not found:\n"
                f"{model_path}"
            )

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False
        )

        self.model = NutriTwinGCN(
            input_features=checkpoint[
                "input_features"
            ],

            hidden_features=checkpoint[
                "hidden_features"
            ],

            output_features=checkpoint[
                "output_features"
            ]
        )

        output_features = checkpoint[
            "output_features"
        ]

        self.prediction_layer = torch.nn.Sequential(

            torch.nn.Linear(
                output_features,
                32
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                32,
                1
            )
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.prediction_layer.load_state_dict(
            checkpoint[
                "prediction_layer_state_dict"
            ]
        )

        self.model.eval()
        self.prediction_layer.eval()

        self.goal = goal


    # ========================================================
    # CHANGE GOAL
    # ========================================================

    def set_goal(self, goal):

        self._load_model(goal)


    # ========================================================
    # CREATE FEATURES
    # ========================================================

    def _create_features(self, nodes):

        features = []

        for node in nodes:

            # ------------------------------------------------
            # Digital Twin
            # ------------------------------------------------

            if node["type"] == "digital_twin":

                features.append([
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ])


            # ------------------------------------------------
            # User Feature
            # ------------------------------------------------

            elif node["type"] == "user_feature":

                value = float(
                    node["value"]
                )

                if node["name"] == "age":
                    value /= 100

                elif node["name"] == "weight":
                    value /= 200

                elif node["name"] == "height":
                    value /= 250

                elif node["name"] == "steps":
                    value /= 20000

                elif node["name"] == "heart_rate":
                    value /= 200

                elif node["name"] == "sleep_hours":
                    value /= 12

                elif node["name"] == "active_calories":
                    value /= 2000

                features.append([
                    value,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ])


            # ------------------------------------------------
            # Food
            # ------------------------------------------------

            elif node["type"] == "food":

                matches = self.foods[
                    self.foods["food"]
                    == node["name"]
                ]

                if matches.empty:

                    raise ValueError(
                        f"Food not found: "
                        f"{node['name']}"
                    )

                food = matches.iloc[0]

                features.append([

                    float(
                        food["calories"]
                    ) / 1000,

                    float(
                        food["protein"]
                    ) / 100,

                    float(
                        food["carbs"]
                    ) / 100,

                    float(
                        food["fat"]
                    ) / 100,

                    float(
                        food["fiber"]
                    ) / 50

                ])


            # ------------------------------------------------
            # Nutrient
            # ------------------------------------------------

            elif node["type"] == "nutrient":

                nutrient_index = {

                    "calories": 0,
                    "protein": 1,
                    "carbs": 2,
                    "fat": 3,
                    "fiber": 4

                }

                vector = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ]

                nutrient = node["name"]

                if nutrient in nutrient_index:

                    vector[
                        nutrient_index[nutrient]
                    ] = 1.0

                features.append(
                    vector
                )

        return torch.tensor(
            features,
            dtype=torch.float32
        )


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(

        self,

        age=22,

        weight=65,

        height=170,

        steps=8000,

        heart_rate=75,

        sleep_hours=7.5,

        active_calories=400,

        top_k=5,

        goal=None

    ):

        # ----------------------------------------------------
        # Change model if goal supplied
        # ----------------------------------------------------

        if goal is not None:

            if goal != self.goal:

                self.set_goal(goal)


        # ----------------------------------------------------
        # Build personalized graph
        # ----------------------------------------------------

        nodes, edges = build_personalized_graph(

            age=age,

            weight=weight,

            height=height,

            steps=steps,

            heart_rate=heart_rate,

            sleep_hours=sleep_hours,

            active_calories=active_calories

        )


        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        x = self._create_features(
            nodes
        )


        # ----------------------------------------------------
        # Node index
        # ----------------------------------------------------

        node_to_index = {

            node["id"]: index

            for index, node
            in enumerate(nodes)

        }


        # ----------------------------------------------------
        # Edges
        # ----------------------------------------------------

        edge_index = []

        for edge in edges:

            source = node_to_index[
                edge["source"]
            ]

            target = node_to_index[
                edge["target"]
            ]

            edge_index.append([
                source,
                target
            ])


        edge_index = torch.tensor(

            edge_index,

            dtype=torch.long

        ).t().contiguous()


        # ----------------------------------------------------
        # Graph
        # ----------------------------------------------------

        graph = Data(

            x=x,

            edge_index=edge_index

        )


        # ----------------------------------------------------
        # GNN inference
        # ----------------------------------------------------

        with torch.no_grad():

            embeddings = self.model(

                graph.x,

                graph.edge_index

            )

            predictions = (

                self.prediction_layer(

                    embeddings

                )

                .squeeze(-1)

            )


        # ----------------------------------------------------
        # Food results
        # ----------------------------------------------------

        results = []

        for index, node in enumerate(nodes):

            if node["type"] == "food":

                results.append({

                    "food": node["name"],

                    "score": float(
                        predictions[index]
                    )

                })


        # ----------------------------------------------------
        # Ranking
        # ----------------------------------------------------

        results.sort(

            key=lambda item:
            item["score"],

            reverse=True

        )


        return results[:top_k]


# ============================================================
# TEST ALL MODELS
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "NutriTwinNet Multi-Goal GNN Predictor"
    )

    print("=" * 60)


    for goal in [

        "Weight Loss",

        "Weight Gain",

        "Maintain Weight"

    ]:

        print()

        print(
            f"GOAL: {goal}"
        )

        print("-" * 60)


        predictor = NutriTwinGNNPredictor(
            goal=goal
        )


        results = predictor.predict(

            age=22,

            weight=65,

            height=170,

            steps=10500,

            heart_rate=76,

            sleep_hours=7.5,

            active_calories=550,

            top_k=5

        )


        for rank, result in enumerate(

            results,

            start=1

        ):

            print(

                f"{rank}. "
                f"{result['food']} "
                f"→ "
                f"{result['score']:.4f}"

            )


    print()

    print("=" * 60)

    print(
        "GNN inference completed successfully."
    )

    print("=" * 60)