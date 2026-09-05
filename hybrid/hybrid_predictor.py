import numpy as np

from gnn.gnn_predictor import NutriTwinGNNPredictor
from rl.rl_predictor import NutriTwinRLPredictor


class NutriTwinHybridPredictor:

    def __init__(
        self,
        goal="Weight Loss",
        gnn_weight=0.6,
        rl_weight=0.4
    ):

        self.goal = goal

        self.gnn_weight = gnn_weight
        self.rl_weight = rl_weight

        self.gnn_predictor = (
            NutriTwinGNNPredictor(
                goal=goal
            )
        )

        self.rl_predictor = (
            NutriTwinRLPredictor(
                goal=goal
            )
        )


    # ========================================================
    # MIN-MAX NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize(values):

        values = np.asarray(
            values,
            dtype=float
        )

        if len(values) == 0:
            return []

        minimum = values.min()
        maximum = values.max()

        if maximum - minimum < 1e-8:

            return np.ones(
                len(values)
            )

        return (
            (values - minimum)
            /
            (maximum - minimum)
        )


    # ========================================================
    # HYBRID PREDICTION
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

        # ----------------------------------------------------
        # GNN candidates
        # ----------------------------------------------------

        gnn_results = (
            self.gnn_predictor.predict(
                age=age,
                weight=weight,
                height=height,
                steps=steps,
                heart_rate=heart_rate,
                sleep_hours=sleep_hours,
                active_calories=active_calories,
                top_k=10
            )
        )

        # ----------------------------------------------------
        # RL candidates
        # ----------------------------------------------------

        rl_results = (
            self.rl_predictor.predict(
                age=age,
                weight=weight,
                height=height,
                steps=steps,
                heart_rate=heart_rate,
                sleep_hours=sleep_hours,
                active_calories=active_calories,
                top_k=10
            )
        )

        # ----------------------------------------------------
        # Create lookup dictionaries
        # ----------------------------------------------------

        gnn_scores = {
            item["food"]: item["score"]
            for item in gnn_results
        }

        rl_scores = {
            item["food"]: item["q_value"]
            for item in rl_results
        }

        # ----------------------------------------------------
        # Combine candidates
        # ----------------------------------------------------

        candidate_foods = sorted(
            set(gnn_scores.keys())
            |
            set(rl_scores.keys())
        )

        if not candidate_foods:
            return []

        # ----------------------------------------------------
        # Raw scores
        # ----------------------------------------------------

        raw_gnn = [
            gnn_scores.get(
                food,
                0.0
            )
            for food in candidate_foods
        ]

        raw_rl = [
            rl_scores.get(
                food,
                0.0
            )
            for food in candidate_foods
        ]

        # ----------------------------------------------------
        # Normalize scores
        # ----------------------------------------------------

        normalized_gnn = self._normalize(
            raw_gnn
        )

        normalized_rl = self._normalize(
            raw_rl
        )

        # ----------------------------------------------------
        # Hybrid score
        # ----------------------------------------------------

        results = []

        for index, food in enumerate(
            candidate_foods
        ):

            hybrid_score = (
                self.gnn_weight
                *
                normalized_gnn[index]
                +
                self.rl_weight
                *
                normalized_rl[index]
            )

            results.append(
                {
                    "food": food,

                    "gnn_score": float(
                        gnn_scores.get(
                            food,
                            0.0
                        )
                    ),

                    "rl_q_value": float(
                        rl_scores.get(
                            food,
                            0.0
                        )
                    ),

                    "gnn_normalized": float(
                        normalized_gnn[index]
                    ),

                    "rl_normalized": float(
                        normalized_rl[index]
                    ),

                    "hybrid_score": float(
                        hybrid_score
                    )
                }
            )

        # ----------------------------------------------------
        # Rank final recommendations
        # ----------------------------------------------------

        results.sort(
            key=lambda item:
                item["hybrid_score"],
            reverse=True
        )

        return results[:top_k]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NutriTwinNet GNN + RL Hybrid Predictor")
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

        predictor = (
            NutriTwinHybridPredictor(
                goal=goal,
                gnn_weight=0.6,
                rl_weight=0.4
            )
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
                f"→ Hybrid: "
                f"{result['hybrid_score']:.4f} "
                f"| GNN: "
                f"{result['gnn_score']:.4f} "
                f"| RL: "
                f"{result['rl_q_value']:.4f}"
            )

    print()
    print("=" * 60)
    print(
        "GNN + RL hybrid inference completed successfully."
    )
    print("=" * 60)