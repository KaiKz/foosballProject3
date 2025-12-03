import numpy as np
import torch

from stable_baselines3 import SAC
from sb3_contrib import TQC

from foosball_multirod_env import FoosballMultiRodEnv


# --------------------------
# Env factory
# --------------------------
def make_env():
    return FoosballMultiRodEnv(
        xml_path="foosball_sim/v2/multirod_foosball.xml",
        render_mode=None,
        episode_length=150,
        frame_skip=2,
    )


# --------------------------
# (Reuse your eval helpers)
# --------------------------
def evaluate_policy(model, env, n_episodes: int = 20, deterministic: bool = True, label: str = "SAC"):
    print(f"\n=== Evaluating {label} policy ===")
    returns, goals, max_ys, contact_counts, goal_steps_all = [], [], [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False

        ep_rew = 0.0
        ep_goals = 0
        contact_steps = 0
        max_ball_y = -1e9
        steps = 0
        goal_step = None

        while not (done or truncated):
            steps += 1
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)

            ep_rew += reward
            ball_pos = info["ball_pos"]
            y = ball_pos[1]
            max_ball_y = max(max_ball_y, y)

            if info.get("had_contact", False):
                contact_steps += 1

            if info.get("is_goal", False):
                ep_goals += 1
                if goal_step is None:
                    goal_step = steps

        returns.append(ep_rew)
        goals.append(ep_goals)
        max_ys.append(max_ball_y)
        contact_counts.append(contact_steps)
        goal_steps_all.append(goal_step if goal_step is not None else -1)

        print(
            f"Ep {ep}: return={ep_rew:.2f}, "
            f"max_ball_y={max_ball_y:.3f}, "
            f"goals={ep_goals}, "
            f"contact_steps={contact_steps}, "
            f"goal_step={goal_step}"
        )

    mean_goal_step = (
        np.mean([g for g in goal_steps_all if g > 0])
        if any(g > 0 for g in goal_steps_all)
        else None
    )

    print(
        f"\n[{label}] mean_return={np.mean(returns):.2f}, "
        f"mean_max_y={np.mean(max_ys):.3f}, "
        f"mean_goals={np.mean(goals):.2f}, "
        f"mean_contact_steps={np.mean(contact_counts):.2f}, "
        f"mean_goal_step={mean_goal_step}"
    )


def evaluate_random(env, n_episodes: int = 20):
    print("\n=== Evaluating RANDOM policy ===")
    returns, goals, max_ys, contact_counts, goal_steps_all = [], [], [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False

        ep_rew = 0.0
        ep_goals = 0
        contact_steps = 0
        max_ball_y = -1e9
        steps = 0
        goal_step = None

        while not (done or truncated):
            steps += 1
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            ep_rew += reward
            y = info["ball_pos"][1]
            max_ball_y = max(max_ball_y, y)

            if info.get("had_contact", False):
                contact_steps += 1

            if info.get("is_goal", False):
                ep_goals += 1
                if goal_step is None:
                    goal_step = steps

        returns.append(ep_rew)
        goals.append(ep_goals)
        max_ys.append(max_ball_y)
        contact_counts.append(contact_steps)
        goal_steps_all.append(goal_step if goal_step is not None else -1)

        print(
            f"Ep {ep}: return={ep_rew:.2f}, "
            f"max_ball_y={max_ball_y:.3f}, "
            f"goals={ep_goals}, "
            f"contact_steps={contact_steps}, "
            f"goal_step={goal_step}"
        )

    mean_goal_step = (
        np.mean([g for g in goal_steps_all if g > 0])
        if any(g > 0 for g in goal_steps_all)
        else None
    )

    print(
        f"\n[RANDOM] mean_return={np.mean(returns):.2f}, "
        f"mean_max_y={np.mean(max_ys):.3f}, "
        f"mean_goals={np.mean(goals):.2f}, "
        f"mean_contact_steps={np.mean(contact_counts):.2f}, "
        f"mean_goal_step={mean_goal_step}"
    )


if __name__ == "__main__":
    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    env = make_env()

    # ------------- SAC -------------
    sac_model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=512,
        learning_starts=1_000,
        buffer_size=100_000,
        train_freq=1,
        gradient_steps=1,
        gamma=0.99,
        tau=0.005,
        device=device,
    )
    sac_model.learn(total_timesteps=50_000)
    sac_model.save("sac_foosball_multirod")

    evaluate_policy(sac_model, env, n_episodes=50, label="SAC")
    evaluate_random(env, n_episodes=50)

#     # ------------- TQC -------------
#     tqc_model = TQC(
#         "MlpPolicy",
#         env,
#         verbose=1,
#         batch_size=512,
#         learning_starts=1_000,
#         buffer_size=100_000,
#         train_freq=1,
#         gradient_steps=1,
#         gamma=0.99,
#         tau=0.005,
#         top_quantiles_to_drop_per_net=2,
#         device=device,
#     )
#     tqc_model.learn(total_timesteps=200_000)
#     tqc_model.save("tqc_foosball_multirod")

#     evaluate_policy(tqc_model, env, n_episodes=50, label="TQC")
