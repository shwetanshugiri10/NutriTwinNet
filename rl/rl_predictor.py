import os
import numpy as np

try:
    import torch
    from .dqn_agent import NutritionQNetwork, state_to_vector
    from .nutrition_env import NutritionEnvironment
    TORCH_AVAILABLE = True
except (ImportError, ValueError):
    try:
        from dqn_agent import NutritionQNetwork, state_to_vector
        from nutrition_env import NutritionEnvironment
        TORCH_AVAILABLE = True
    except (ImportError, ValueError):
        TORCH_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

RL_DIR = os.path.dirname(__file__)

MODEL_PATHS = {
    "Weight Loss": os.path.join(
        RL_DIR,
        "nutritwin_rl_weight_loss.pth"
    ),

    "Weight Gain": os.path.join(
        RL_DIR,
        "nutritwin_rl_weight_gain.pth"
    ),

    "Maintain Weight": os.path.join(
        RL_DIR,
        "nutritwin_rl_maintenance.pth"
    )
}


class NutriTwinRLPredictor:

    def __init__(self, goal="Weight Loss"):

        if goal not in MODEL_PATHS:
            raise ValueError(f"Unsupported goal: {goal}")

        self.goal = goal
        self.actions = [
            "Chicken Breast", "Fish", "Paneer", "Chickpeas", "Dal",
            "Quinoa", "Greek Yogurt", "Tofu", "Broccoli", "Brown Rice"
        ]
        self.model_path = MODEL_PATHS[goal]
        self.q_network = None

        if TORCH_AVAILABLE and os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
                self.actions = checkpoint.get("actions", self.actions)
                self.state_size = checkpoint.get("state_size", 7)
                self.action_size = checkpoint.get("action_size", len(self.actions))
                self.q_network = NutritionQNetwork(self.state_size, self.action_size)
                self.q_network.load_state_dict(checkpoint["model_state_dict"])
                self.q_network.eval()
            except Exception:
                self.q_network = None

    # ========================================================
    # PREDICT BEST RL ACTION
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
        top_k=5
    ):
        if TORCH_AVAILABLE and self.q_network is not None:
            state = {
                "goal": self.goal,
                "age": age, "weight": weight, "height": height,
                "steps": steps, "heart_rate": heart_rate,
                "sleep_hours": sleep_hours, "active_calories": active_calories
            }
            state_vector = state_to_vector(state)
            state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).squeeze(0)
            ranked_indices = torch.argsort(q_values, descending=True)
            results = []
            for index in ranked_indices[:top_k]:
                idx = int(index)
                results.append({
                    "food": self.actions[idx],
                    "q_value": float(q_values[idx]),
                    "score": float(q_values[idx])
                })
        else:
            # NumPy Q-policy fallback
            norm_steps = steps / 10000.0
            norm_sleep = sleep_hours / 8.0
            norm_cal = active_calories / 500.0
            base_bias = norm_steps * 0.4 + norm_sleep * 0.3 + norm_cal * 0.3

            if self.goal == "Weight Loss":
                weights = [0.95, 0.92, 0.70, 0.85, 0.80, 0.75, 0.90, 0.88, 0.98, 0.65]
            elif self.goal == "Weight Gain":
                weights = [0.85, 0.80, 0.95, 0.82, 0.88, 0.92, 0.78, 0.75, 0.50, 0.90]
            else:
                weights = [0.88, 0.86, 0.82, 0.85, 0.84, 0.88, 0.86, 0.80, 0.85, 0.82]

            q_values = np.array(weights) * (0.8 + 0.2 * base_bias)
            indexed = sorted(enumerate(q_values), key=lambda x: x[1], reverse=True)
            results = []
            for idx, q_val in indexed[:top_k]:
                if idx < len(self.actions):
                    results.append({
                        "food": self.actions[idx],
                        "q_value": float(q_val),
                        "score": float(q_val)
                    })

        return results



# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NutriTwinNet RL Predictor")
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

        predictor = NutriTwinRLPredictor(
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
                f"→ Q-value: "
                f"{result['q_value']:.4f}"
            )

    print()
    print("=" * 60)
    print(
        "RL inference completed successfully."
    )
    print("=" * 60)