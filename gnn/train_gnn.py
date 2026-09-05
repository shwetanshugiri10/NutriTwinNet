import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
import pandas as pd

from graph_builder import build_personalized_graph
from gcn_model import NutriTwinGCN


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 300
LEARNING_RATE = 0.01

GOAL = "Weight Loss"


# ============================================================
# LOAD FOOD DATA
# ============================================================

foods = pd.read_csv("data/foods.csv")


# ============================================================
# BUILD PERSONALIZED GRAPH
# ============================================================

nodes, edges = build_personalized_graph(
    age=22,
    weight=65,
    height=170,
    steps=10500,
    heart_rate=76,
    sleep_hours=7.5,
    active_calories=550
)


# ============================================================
# CREATE NODE FEATURES
# ============================================================

features = []

for node in nodes:

    # --------------------------------------------------------
    # DIGITAL TWIN
    # --------------------------------------------------------

    if node["type"] == "digital_twin":

        features.append([
            1.0,
            0.0,
            0.0,
            0.0,
            0.0
        ])

    # --------------------------------------------------------
    # USER FEATURES
    # --------------------------------------------------------

    elif node["type"] == "user_feature":

        value = float(node["value"])

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

    # --------------------------------------------------------
    # FOOD FEATURES
    # --------------------------------------------------------

    elif node["type"] == "food":

        food_name = node["name"]

        food = foods[
            foods["food"] == food_name
        ].iloc[0]

        calories = float(food["calories"]) / 1000
        protein = float(food["protein"]) / 100
        carbs = float(food["carbs"]) / 100
        fat = float(food["fat"]) / 100
        fiber = float(food["fiber"]) / 50

        features.append([
            calories,
            protein,
            carbs,
            fat,
            fiber
        ])

    # --------------------------------------------------------
    # NUTRIENT NODES
    # --------------------------------------------------------

    elif node["type"] == "nutrient":

        nutrient_index = {
            "calories": 0,
            "protein": 1,
            "carbs": 2,
            "fat": 3,
            "fiber": 4
        }

        vector = [0.0] * 5

        vector[
            nutrient_index[node["name"]]
        ] = 1.0

        features.append(vector)


# ============================================================
# FEATURE TENSOR
# ============================================================

x = torch.tensor(
    features,
    dtype=torch.float
)


# ============================================================
# CREATE EDGE INDEX
# ============================================================

node_to_index = {
    node["id"]: index
    for index, node in enumerate(nodes)
}

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


# ============================================================
# GRAPH
# ============================================================

graph = Data(
    x=x,
    edge_index=edge_index
)


print("=" * 60)
print("NutriTwinNet Improved GNN Training")
print("=" * 60)

print(f"Nodes: {len(nodes)}")
print(f"Edges: {len(edges)}")
print(f"Input Features: {x.shape[1]}")
print(f"Goal: {GOAL}")


# ============================================================
# CREATE TARGET SCORES
# ============================================================

targets = torch.zeros(
    len(nodes),
    dtype=torch.float
)

food_indices = []


for index, node in enumerate(nodes):

    if node["type"] == "food":

        food_indices.append(index)

        food = foods[
            foods["food"] == node["name"]
        ].iloc[0]

        calories = float(food["calories"])
        protein = float(food["protein"])
        fiber = float(food["fiber"])
        fat = float(food["fat"])

        # ----------------------------------------------------
        # WEIGHT LOSS
        # ----------------------------------------------------

        if GOAL == "Weight Loss":

            score = (
                protein * 3
                + fiber * 2
                - fat * 0.5
                - calories * 0.01
            )

        # ----------------------------------------------------
        # WEIGHT GAIN
        # ----------------------------------------------------

        elif GOAL == "Weight Gain":

            score = (
                protein * 3
                + calories * 0.01
                + fat * 0.2
            )

        # ----------------------------------------------------
        # MAINTENANCE
        # ----------------------------------------------------

        else:

            score = (
                protein * 2
                + fiber * 2
                - abs(calories - 200) * 0.01
            )

        targets[index] = score


# ============================================================
# NORMALIZE TARGETS
# ============================================================

food_targets = targets[food_indices]

minimum = food_targets.min()
maximum = food_targets.max()

targets[food_indices] = (
    (food_targets - minimum)
    /
    (maximum - minimum + 1e-8)
)


# ============================================================
# MODEL
# ============================================================

model = NutriTwinGCN(
    input_features=5,
    hidden_features=32,
    output_features=16
)


# ============================================================
# PREDICTION HEAD
# ============================================================

prediction_layer = torch.nn.Sequential(
    torch.nn.Linear(16, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1)
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = Adam(
    list(model.parameters())
    +
    list(prediction_layer.parameters()),
    lr=LEARNING_RATE
)


# ============================================================
# TRAIN
# ============================================================

for epoch in range(EPOCHS):

    model.train()
    prediction_layer.train()

    optimizer.zero_grad()

    embeddings = model(
        graph.x,
        graph.edge_index
    )

    predictions = prediction_layer(
        embeddings
    ).squeeze()

    loss = F.mse_loss(
        predictions[food_indices],
        targets[food_indices]
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 25 == 0:

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {loss.item():.6f}"
        )


# ============================================================
# EVALUATION
# ============================================================

model.eval()
prediction_layer.eval()

with torch.no_grad():

    embeddings = model(
        graph.x,
        graph.edge_index
    )

    predictions = prediction_layer(
        embeddings
    ).squeeze()


# ============================================================
# FOOD RANKING
# ============================================================

results = []

for index in food_indices:

    results.append({
        "food": nodes[index]["name"],
        "score": float(
            predictions[index]
        ),
        "target": float(
            targets[index]
        )
    })


results.sort(
    key=lambda item: item["score"],
    reverse=True
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PERSONALIZED GNN FOOD RANKING")
print("=" * 60)

for rank, item in enumerate(
    results,
    start=1
):

    print(
        f"{rank:02d}. "
        f"{item['food']:<20} "
        f"Predicted: {item['score']:.4f} | "
        f"Target: {item['target']:.4f}"
    )


# ============================================================
# CHECK WHETHER MODEL LEARNED DIFFERENCES
# ============================================================

scores = [
    item["score"]
    for item in results
]

score_range = max(scores) - min(scores)

print("\n" + "=" * 60)

print(
    f"Prediction Range: {score_range:.6f}"
)

if score_range > 0.01:

    print(
        "✅ GNN is producing different food scores."
    )

else:

    print(
        "⚠️ Food scores are still very similar."
    )

print("=" * 60)
print("GNN training completed.")
print("=" * 60)


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "prediction_layer_state_dict": prediction_layer.state_dict(),
        "goal": GOAL,
        "input_features": 5,
        "hidden_features": 32,
        "output_features": 16
    },
    "gnn/nutritwin_gnn.pth"
)

print("\n✅ Trained GNN saved to:")
print("gnn/nutritwin_gnn.pth")