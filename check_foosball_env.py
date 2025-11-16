from stable_baselines3.common.env_checker import check_env
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv

def main():
    env = FoosballEnv(
        antagonist_model=None,
        play_until_goal=False,
        verbose_mode=False,
        # debug_free_ball=False  # <-- use real contacts for training
    )
    check_env(env, warn=True)

if __name__ == "__main__":
    main()
