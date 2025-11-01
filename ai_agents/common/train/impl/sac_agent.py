from stable_baselines3 import SAC
from ai_agents.common.train.interface.foosball_agent import FoosballAgent
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from tqdm import tqdm
import os


class TqdmCallback(BaseCallback):
    """
    Callback for stable_baselines3 to show progress with tqdm.
    """
    def __init__(self, total_timesteps=None, verbose=0):
        super().__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps
    
    def _on_training_start(self):
        # Get total_timesteps from locals which is set by stable_baselines3
        # Fallback to the one passed in constructor if locals doesn't have it
        total = self.locals.get('total_timesteps', self.total_timesteps)
        if total:
            self.pbar = tqdm(total=total, desc="Training", unit="step")
        else:
            self.pbar = tqdm(desc="Training", unit="step")
    
    def _on_step(self):
        if self.pbar is not None:
            self.pbar.update(1)
        return True
    
    def _on_training_end(self):
        if self.pbar is not None:
            self.pbar.close()


class SACFoosballAgent(FoosballAgent):
    def __init__(self, id:int, env=None, log_dir='./logs', model_dir='./models', policy_kwargs = dict(net_arch=[3000, 3000, 3000, 3000, 3000, 3000, 3000])):
        self.env = env
        self.model = None
        self.id = id
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.id_subdir = f'{model_dir}/{id}'
        self.policy_kwargs = policy_kwargs

    def get_id(self):
        return self.id

    def initialize_agent(self):
        try:
            self.load()
        except Exception as e:
            print(f"Agent {self.id} could not load model. Initializing new model.")
            self.model = SAC('MlpPolicy', self.env, policy_kwargs=self.policy_kwargs, device='cuda', buffer_size=1000000)
        print(f"Agent {self.id} initialized.")

    def predict(self, observation, deterministic=False):
        if self.model is None:
            raise ValueError("Model has not been initialized.")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def learn(self, total_timesteps):
        if self.model is None:
            self.model = SAC('MlpPolicy', self.env, policy_kwargs=self.policy_kwargs, buffer_size=1000000)
        callback = self.create_callback(self.env, total_timesteps)
        tb_log_name = f'sac_{self.id}'
        self.model.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name=tb_log_name)

    def create_callback(self, env, total_timesteps=None, show_progress=True):
        eval_callback = EvalCallback(
            env,
            best_model_save_path=self.id_subdir + '/sac/best_model',
            log_path=self.log_dir,
            eval_freq=3000,
            n_eval_episodes=10,
            render=False,
            deterministic=True,
        )
        if show_progress and os.getenv('DISPLAY_PROGRESS', '1') != '0':
            callback_list = CallbackList([TqdmCallback(total_timesteps=total_timesteps), eval_callback])
            return callback_list
        return eval_callback

    def save(self):
        self.model.save(self.id_subdir + '/sac/best_model')

    def load(self):
        self.model = SAC.load(self.id_subdir + '/sac/best_model/best_model.zip')
        print(f"Agent {self.id} loaded model from {self.id_subdir}/sac/best_model/best_model.zip")

    def change_env(self, env):
        self.env = env
        self.model.set_env(env)