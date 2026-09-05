import random


class NutritionEnvironment:

    def __init__(self):

        self.goals = [
            "Weight Loss",
            "Weight Gain",
            "Maintain Weight"
        ]

        self.actions = [
            "Chicken Breast",
            "Fish",
            "Paneer",
            "Chickpeas",
            "Dal",
            "Quinoa",
            "Greek Yogurt",
            "Tofu",
            "Broccoli",
            "Brown Rice"
        ]

        self.current_goal = None
        self.current_state = None

        self.reset()


    def reset(
        self,
        goal="Weight Loss",
        user_profile=None
    ):

        self.current_goal = goal

        if user_profile is None:

            user_profile = {
                "age": 22,
                "weight": 65,
                "height": 170,
                "steps": 8000,
                "heart_rate": 75,
                "sleep_hours": 7.5,
                "active_calories": 400
            }

        self.current_state = user_profile.copy()

        return self._get_state()


    def _get_state(self):

        return {
            "goal": self.current_goal,
            **self.current_state
        }


    def step(self, action):

        if action not in self.actions:

            raise ValueError(
                f"Invalid action: {action}"
            )

        reward = self._calculate_reward(
            action
        )

        next_state = self._simulate_feedback()

        done = True

        return (
            next_state,
            reward,
            done
        )


    def _calculate_reward(self, food):

        # ------------------------------------------
        # Simplified prototype reward function
        # ------------------------------------------

        high_protein = [
            "Chicken Breast",
            "Fish",
            "Paneer",
            "Chickpeas",
            "Dal",
            "Greek Yogurt",
            "Tofu"
        ]

        high_energy = [
            "Paneer",
            "Chicken Breast",
            "Fish",
            "Quinoa",
            "Brown Rice"
        ]

        balanced = [
            "Dal",
            "Chickpeas",
            "Quinoa",
            "Chicken Breast",
            "Fish"
        ]

        reward = 0.0

        if self.current_goal == "Weight Loss":

            if food in balanced:
                reward += 1.0

            if food in high_protein:
                reward += 0.5

        elif self.current_goal == "Weight Gain":

            if food in high_energy:
                reward += 1.0

            if food in high_protein:
                reward += 0.5

        elif self.current_goal == "Maintain Weight":

            if food in balanced:
                reward += 1.0

            if food in high_protein:
                reward += 0.25

        return reward


    def _simulate_feedback(self):

        next_state = self.current_state.copy()

        # Small simulated wearable changes
        next_state["steps"] += random.randint(
            -500,
            500
        )

        next_state["heart_rate"] += random.randint(
            -2,
            2
        )

        next_state["sleep_hours"] += random.uniform(
            -0.2,
            0.2
        )

        next_state["active_calories"] += random.randint(
            -30,
            30
        )

        return {
            "goal": self.current_goal,
            **next_state
        }


if __name__ == "__main__":

    print("=" * 60)
    print("NutriTwinNet Reinforcement Learning Environment")
    print("=" * 60)

    env = NutritionEnvironment()

    state = env.reset(
        goal="Weight Gain"
    )

    print()
    print("Initial State:")
    print(state)

    action = "Paneer"

    next_state, reward, done = env.step(
        action
    )

    print()
    print("Action:")
    print(action)

    print()
    print("Reward:")
    print(reward)

    print()
    print("Next State:")
    print(next_state)

    print()
    print("Episode Done:")
    print(done)

    print()
    print("=" * 60)
    print(
        "RL environment created successfully."
    )
    print("=" * 60)