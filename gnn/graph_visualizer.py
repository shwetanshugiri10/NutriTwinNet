import plotly.graph_objects as go
import numpy as np

class GNNGraphVisualizer:
    """
    Renders an interactive 2D/3D heterogeneous graph visualization
    showing User Twin, Biometric nodes, Nutrient nodes, and Top Ranked Foods.
    """
    def __init__(self):
        pass

    def build_plotly_subgraph(self, user_name="User", top_foods=None, twin_score=85.0):
        if top_foods is None:
            top_foods = [
                {"food": "Salmon Fillet", "category": "Protein", "score": 0.98},
                {"food": "Quinoa Bowl", "category": "Grain", "score": 0.94},
                {"food": "Spinach Salad", "category": "Vegetable", "score": 0.91},
                {"food": "Greek Yogurt", "category": "Dairy", "score": 0.89},
                {"food": "Almonds", "category": "Nuts", "score": 0.86}
            ]

        # Central User Twin Node at (0, 0)
        node_x = [0.0]
        node_y = [0.0]
        node_names = [f"🧬 Digital Twin: {user_name}"]
        node_colors = ["#10B981"]
        node_sizes = [32]
        node_symbols = ["diamond"]

        # Biometric Feature Nodes (Inner Ring)
        biometrics = ["Steps", "Resting HR", "Sleep", "Active Burn", "BMR"]
        r_bio = 0.45
        for i, bio in enumerate(biometrics):
            angle = 2 * np.pi * i / len(biometrics)
            x = r_bio * np.cos(angle)
            y = r_bio * np.sin(angle)
            node_x.append(x)
            node_y.append(y)
            node_names.append(f"📊 {bio}")
            node_colors.append("#3B82F6")
            node_sizes.append(18)
            node_symbols.append("circle")

        # Nutrient Attribute Nodes (Middle Ring)
        nutrients = ["Protein", "Carbs", "Fat", "Fiber", "Micros"]
        r_nut = 0.85
        for i, nut in enumerate(nutrients):
            angle = (2 * np.pi * i / len(nutrients)) + (np.pi / len(nutrients))
            x = r_nut * np.cos(angle)
            y = r_nut * np.sin(angle)
            node_x.append(x)
            node_y.append(y)
            node_names.append(f"⚡ {nut}")
            node_colors.append("#F59E0B")
            node_sizes.append(20)
            node_symbols.append("square")

        # Top Recommended Food Nodes (Outer Ring)
        r_food = 1.35
        for i, item in enumerate(top_foods[:6]):
            angle = 2 * np.pi * i / min(len(top_foods), 6)
            x = r_food * np.cos(angle)
            y = r_food * np.sin(angle)
            node_x.append(x)
            node_y.append(y)
            node_names.append(f"🥗 {item['food']} ({item.get('score', 0.9):.2f})")
            node_colors.append("#EC4899")
            node_sizes.append(24)
            node_symbols.append("hexagon")

        # Edges
        edge_x = []
        edge_y = []

        # 1. Twin -> Biometrics
        for i in range(1, len(biometrics) + 1):
            edge_x.extend([node_x[0], node_x[i], None])
            edge_y.extend([node_y[0], node_y[i], None])

        # 2. Biometrics -> Nutrients
        for i in range(1, len(biometrics) + 1):
            nut_idx = len(biometrics) + 1 + (i % len(nutrients))
            edge_x.extend([node_x[i], node_x[nut_idx], None])
            edge_y.extend([node_y[i], node_y[nut_idx], None])

        # 3. Nutrients -> Foods
        food_start = len(biometrics) + len(nutrients) + 1
        for f_idx in range(food_start, len(node_x)):
            # Connect food to 2 adjacent nutrients
            nut_1 = len(biometrics) + 1 + (f_idx % len(nutrients))
            edge_x.extend([node_x[nut_1], node_x[f_idx], None])
            edge_y.extend([node_y[nut_1], node_y[f_idx], None])

        fig = go.Figure()
        
        # Edge lines
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(color="rgba(255, 255, 255, 0.25)", width=1.5),
            hoverinfo="none",
            showlegend=False
        ))

        # Node points
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=[name.split(":")[0] if ":" in name else name for name in node_names],
            textposition="top center",
            textfont=dict(size=10, color="#FFFFFF"),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                symbol=node_symbols,
                line=dict(width=1.5, color="#FFFFFF")
            ),
            hovertext=node_names,
            hoverinfo="text",
            showlegend=False
        ))

        fig.update_layout(
            title="🕸️ Heterogeneous GNN Message-Passing Subgraph Topology",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig
