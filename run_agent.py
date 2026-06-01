import json
from sb3_contrib import MaskablePPO
from timetable_env import TimetableEnv

def run_inference():

    # Load model and config
    model = MaskablePPO.load("ppo_timetable_final.zip")

    with open("config.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # (Assume these are already built as in training pipeline)
    # teachers, rooms, student_groups, courses, total_slots, periods_per_day, days_per_week, core_map
    # ... preprocessing steps here ...

    # Build environment (same structure as training phase)
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
    done = False

    total_reward = 0
    success_count = 0
    total_steps = 0

    while not done:
        action_masks = env.get_action_mask()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_steps += 1

        if "error" not in info:
            success_count += 1

        done = terminated or truncated

    total_sessions = len(env.sessions)

    # Evaluation metrics
    success_rate = (success_count / total_sessions) * 100 if total_sessions else 0
    avg_reward = total_reward / total_steps if total_steps else 0

    print("\n===== EVALUATION RESULTS =====")
    print(f"Total Reward        : {total_reward:.2f}")
    print(f"Average Reward      : {avg_reward:.4f}")
    print(f"Total sessions       : {total_sessions}")
    print(f"Successfully Placed : {success_count}")
    print(f"Success Rate        : {success_rate:.2f}%")
    print("================================")

if __name__ == "__main__":
    run_inference()