import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class ETACallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_rollout_end(self):
        elapsed = time.time() - self.start_time
        done    = self.num_timesteps
        speed   = done / elapsed if elapsed > 0 else 1
        remaining = (self.total_timesteps - done) / speed
        m, s = divmod(int(remaining), 60)
        h, m = divmod(m, 60)
        print(f"  ETA: {h:02d}:{m:02d}:{s:02d}  ({done}/{self.total_timesteps} steps)")

    def _on_step(self):
        return True


class PPOAgent:
    def __init__(self, env, seed=0):
        self.env = env
        self.model = PPO(
            'MlpPolicy', env, seed=seed, verbose=1, device='cpu',
        )

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps,
                         callback=ETACallback(total_timesteps))

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path)

    def predict(self, observation):
        action, _states = self.model.predict(observation)
        return action
