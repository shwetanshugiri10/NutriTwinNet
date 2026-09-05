import torch
from torch_geometric.data import Data

from graph_builder import build_personalized_graph
from gcn_model import NutriTwinGCN


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

print("=" * 60)
print("NutriTwinNet GNN Execution")
print("=" * 60)

print(f"Nodes: {len(nodes)}")
print(f"Edges: {len(edges)}")


# ============================================================
# NODE FEATURES
# ============================================================

features = []

for node in nodes:

    if node["type"] == "digital_twin":

        features.append([
            1.0,
            0.0,
            0.0,
            0.0,
            0.0
        ])

    elif node["type"] == "user_feature":

        value = float(node["value"])

        # Normalize large values
        if node["name"] == "age":
            value = value / 100

        elif node["name"] == "weight":
            value = value / 200

        elif node["name"] == "height":
            value = value / 250

        elif node["name"] == "steps":
            value = value / 20000

        elif node["name"] == "heart_rate":
            value = value / 200

        elif node["name"] == "sleep_hours":
            value = value / 12

        elif node["name"] == "active_calories":
            value = value / 2000

        features.append([
            value,
            0.0,
            0.0,
            0.0,
            0.0
        ])

    elif node["type"] == "food":

        features.append([
            0.0,
            1.0,
            0.0,
            0.0,
            0.0
        ])

    elif node["type"] == "nutrient":

        features.append([
            0.0,
            0.0,
            1.0,
            0.0,
            0.0
        ])


# Convert features to PyTorch tensor

x = torch.tensor(
    features,
    dtype=torch.float
)


# ============================================================
# EDGE INDEX
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
# CREATE PYTORCH GEOMETRIC GRAPH
# ============================================================

graph = Data(
    x=x,
    edge_index=edge_index
)

print("\nPyTorch Geometric Graph:")
print(graph)


# ============================================================
# CREATE GCN MODEL
# ============================================================

model = NutriTwinGCN(
    input_features=5,
    hidden_features=32,
    output_features=16
)


# ============================================================
# RUN GNN
# ============================================================

model.eval()

with torch.no_grad():

    embeddings = model(
        graph.x,
        graph.edge_index
    )


# ============================================================
# RESULTS
# ============================================================

print("\nGNN Output:")
print(
    f"Embedding Shape: {embeddings.shape}"
)

print(
    "\nExpected:"
    f" [{len(nodes)}, 16]"
)


# ============================================================
# DIGITAL TWIN EMBEDDING
# ============================================================

user_embedding = embeddings[0]

print("\nDigital Twin Embedding:")

print(
    user_embedding
)


print("\n" + "=" * 60)
print("GNN execution completed successfully.")
print("=" * 60)