from stable_baselines3 import PPO

class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.model = PPO('MlpPolicy', env, verbose=1, device='cpu') # CPU is better for MLPs, GPU is better for CNNs/RNNs

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path)

    def predict(self, observation):
        action, _states = self.model.predict(observation)
        return action
