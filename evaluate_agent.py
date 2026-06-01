import numpy as np
import os
import json
from sb3_contrib import MaskablePPO
from timetable_env import TimetableEnv


def evaluate_model(model_path, n_episodes=5):

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    with open("config.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n" + "=" * 50)
    print("INITIALIZING EVALUATION")
    print("=" * 50)

    model = MaskablePPO.load(model_path)
    print(f"Loaded model: {os.path.basename(model_path)}")

    history = {
        "fill_rates": [],
        "rewards": [],
        "failed": []
    }

    for episode in range(n_episodes):

        # Environment initialization (same structure as training phase)
        env = TimetableEnv(
            teachers=teachers,
            rooms=rooms,
            student_groups=student_groups,
            courses=courses,
            days_per_week=days_per_week,
            n_slots=total_slots,
            periods_per_day=periods_per_day,
            core_subjects_map=core_map
        )

        obs, _ = env.reset()
        terminated = False
        truncated = False

        episode_reward = 0
        success_count = 0

        while not (terminated or truncated):

            action_masks = env.get_action_mask()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward

            if "error" not in info:
                success_count += 1

        total = len(env.courses)
        fill_rate = (success_count / total) * 100 if total else 0

        history["fill_rates"].append(fill_rate)
        history["rewards"].append(episode_reward)
        history["failed"].append(total - success_count)

        print(f"Episode {episode+1:02d} | Success Rate: {fill_rate:6.2f}% | Reward: {episode_reward:8.2f}")

    print("\n" + "=" * 20 + " FINAL RESULTS " + "=" * 20)
    print(f"Mean Success Rate : {np.mean(history['fill_rates']):.2f}%")
    print(f"Best Success Rate : {np.max(history['fill_rates']):.2f}%")
    print(f"Worst Success Rate: {np.min(history['fill_rates']):.2f}%")
    print(f"Avg Reward        : {np.mean(history['rewards']):.2f}")
    print(f"Avg Failed        : {np.mean(history['failed']):.1f}")
    print("=" * 55)
    env.render()
if __name__ == "__main__":
    evaluate_model(".\\ppo_timetable_final.zip", n_episodes=5)