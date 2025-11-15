# run_tqc_viewer.py
from sb3_contrib import TQC
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv

MODEL_PATH = "/Users/kaikaizhang/Downloads/foosballpart2/models/0/tqc/best_model/best_model_fixed.zip"

def main():
    env = FoosballEnv(antagonist_model=None)  # this should open a window when you call env.render()
    model = TQC.load(MODEL_PATH, env=env)

    obs, _ = env.reset()
    done = False
    step = 0

    while not done:
        step += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # This should draw the current frame in a GL window
        env.render()

        done = terminated or truncated

    env.close()

if __name__ == "__main__":
    main()
