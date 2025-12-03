import torch
import time

from stable_baselines3 import SAC
from foosball_single_rod_env import FoosballSingleRodEnv


def make_env_for_view():
    return FoosballSingleRodEnv(
        xml_path="foosball_sim/v2/multirod_foosball.xml",
        render_mode="human",   # <-- key difference
        episode_length=150,
        frame_skip=2,
        ball_body_name="ball_body",
        player_body_name="player_1",
        rod_joint_name="rod_1_slide",
        rod_actuator_name="rod_1_motor",
    )


if __name__ == "__main__":
    # device not super important for visualization, but okay to keep MPS
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using device: mps")
    else:
        device = "cpu"
        print("Using device: cpu")

    env = make_env_for_view()

    # Load your trained model (adjust name if you saved differently)
    model = SAC.load("sac_foosball_single_rod_harder", device=device)

    n_episodes = 10
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_rew = 0.0
        t = 0

        while not (done or truncated):
            t += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_rew += reward

            # render current state
            env.render()
            # slow down so you can see it (optional)
            time.sleep(0.01)

        print(f"[SAC] Episode {ep}: return={ep_rew:.2f}, steps={t}")

    env.close()
