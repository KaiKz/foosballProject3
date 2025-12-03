import os
import numpy as np
import torch

from stable_baselines3 import SAC
from sb3_contrib import TQC

from foosball_single_rod_env import FoosballSingleRodEnv


# -----------------------------------------------------------
# Env factory
# -----------------------------------------------------------
def make_env():
    return FoosballSingleRodEnv(
        xml_path="foosball_sim/v2/minimal_foosball_stage1.xml",
        render_mode=None,
        episode_length=150,         # must match env default (or override)
        frame_skip=2,
        ball_body_name="ball_body",
        player_body_name="player_1",
        rod_joint_name="rod_1_slide",
        rod_actuator_name="rod_1_motor",
    )


# -----------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------
def evaluate_policy(model, env, n_episodes: int = 5, deterministic: bool = True, label: str = "SAC"):
    print(f"\n=== Evaluating {label} policy ===")
    returns = []
    goals = []
    max_ys = []
    contact_counts = []
    goal_steps_all = []

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

    print(f"\n[{label}] mean_return={np.mean(returns):.2f}, "
          f"mean_max_y={np.mean(max_ys):.3f}, "
          f"mean_goals={np.mean(goals):.2f}, "
          f"mean_contact_steps={np.mean(contact_counts):.2f}, "
          f"mean_goal_step={np.mean([g for g in goal_steps_all if g > 0]) if any(g > 0 for g in goal_steps_all) else None}")


def evaluate_random(env, n_episodes: int = 5):
    print("\n=== Evaluating RANDOM policy ===")
    returns = []
    goals = []
    max_ys = []
    contact_counts = []
    goal_steps_all = []

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

    print(f"\n[RANDOM] mean_return={np.mean(returns):.2f}, "
          f"mean_max_y={np.mean(max_ys):.3f}, "
          f"mean_goals={np.mean(goals):.2f}, "
          f"mean_contact_steps={np.mean(contact_counts):.2f}, "
          f"mean_goal_step={np.mean([g for g in goal_steps_all if g > 0]) if any(g > 0 for g in goal_steps_all) else None}")


def scripted_policy(obs: np.ndarray) -> np.ndarray:
    """
    Very simple heuristic:
      - Move rod toward ball_y with moderate gain.
      - Clamp to [-1, 1].
    """
    ball_y = obs[1]
    rod_y = obs[6]

    target = ball_y  # try to align rod with ball
    error = target - rod_y
    action = 3.0 * error     # proportional control
    action = np.clip(action, -1.0, 1.0)
    return np.array([action], dtype=np.float32)


def evaluate_scripted(env, n_episodes: int = 5):
    print("\n=== Evaluating SCRIPTED policy ===")
    returns = []
    goals = []
    max_ys = []
    contact_counts = []
    goal_steps_all = []

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
            action = scripted_policy(obs)
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

    print(f"\n[SCRIPTED] mean_return={np.mean(returns):.2f}, "
          f"mean_max_y={np.mean(max_ys):.3f}, "
          f"mean_goals={np.mean(goals):.2f}, "
          f"mean_contact_steps={np.mean(contact_counts):.2f}, "
          f"mean_goal_step={np.mean([g for g in goal_steps_all if g > 0]) if any(g > 0 for g in goal_steps_all) else None}")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

if __name__ == "__main__":
    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using device: mps")
    else:
        device = "cpu"
        print("Using device: cpu")

    env = make_env()

    # ----------------- SAC TRAINING -----------------
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
    sac_model.learn(total_timesteps=200_000)
    sac_model.save("sac_foosball_single_rod_harder")

    # ----------------- EVALUATION -----------------
    evaluate_policy(sac_model, env, n_episodes=100, label="SAC")
    evaluate_random(env, n_episodes=100)
    evaluate_scripted(env, n_episodes=100)

    # ----------------- TQC TRAINING -----------------
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
#     tqc_model.learn(total_timesteps=50_000)
#     tqc_model.save("tqc_foosball_single_rod_harder")

#     # Optional: evaluate TQC too
#     evaluate_policy(tqc_model, env, n_episodes=5, label="TQC")
