'''
Portfolio management environment for reinforcement learning agents.
This environment simulates a financial market where an agent can allocate its portfolio among different assets. 
The agent receives observations about the market and takes actions to adjust its portfolio allocation. 
The goal is to maximize the cumulative return over time while managing risk.
The reward is based on the Sharpe ratio, which considers both the return and the volatility of the portfolio and a penalty for the volume of allocations.
The portfolio has 6 assets and cash, and the agent can allocate its portfolio among these assets and cash.
The starting vector is [0, 0, 0, 0, 0, 0, 1], which means that the agent starts with all its capital in cash.
The action space is a continuous space where the agent can allocate its portfolio among the 6 assets and cash. 
The action is a vector of length 7, where each element represents the allocation to a specific asset or cash. 
The sum of the allocations must be equal to 1, which means that the agent must allocate all its capital among the assets and cash. 
The action space is defined as a Box space with shape (7,) and bounds [0, 1] for each element.
Each step, represents a day in the market, and the agent receives a new observation of the market state,
which includes the log-returns of the assets, open - close ranges and maximun - minimum ranges, the Vix value and a prediction of the values.
'''
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import matplotlib.pyplot as plt

from IPM.ipm import IPMModule

class PortfolioEnv(gym.Env):
    def __init__(self, data, transaction_cost: float = 0.001, ipm_module=None, debug=False, debug_every=200, window_size=5, rebalance_freq=5, episode_weeks=52, reward_scale=1.0):
        super(PortfolioEnv, self).__init__()
        self.data = data
        self.tc   = transaction_cost
        self.ipm_module = ipm_module
        self.ipm_dim = ipm_module.m if ipm_module is not None else 0
        self.debug = debug
        self.debug_every = debug_every
        self._debug_buf = []
        self._episode_idx = 0
        self._start_order = np.arange(0)
        self.reward_scale = reward_scale
        self.current_step = 0
        self.num_assets = 6  # 6 assets + cash
        self.window_size = window_size
        self.rebalance_freq = rebalance_freq
        self.episode_days = episode_weeks * rebalance_freq if episode_weeks is not None else None
        self.features_per_day = 19  # 18 features + VIX
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets + 1,), dtype=np.float32)
        # window_size log-returns por activo + features del día actual + ipm + pesos
        obs_dim = self.num_assets * window_size + self.features_per_day + self.ipm_dim + self.num_assets + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.weights = np.zeros(self.num_assets + 1, dtype=np.float32)
        self.weights[-1] = 1.0  # todo en cash al inicio

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.debug and self._debug_buf:
            arr = {k: np.array([x[k] for x in self._debug_buf]) for k in self._debug_buf[0]}
            print(
                "[debug] port_ret mean/min/max:",
                arr["port_ret"].mean(), arr["port_ret"].min(), arr["port_ret"].max()
            )
            print(
                "[debug] reward mean/min/max:",
                arr["reward"].mean(), arr["reward"].min(), arr["reward"].max()
            )
            print(
                "[debug] turnover mean/min/max:",
                arr["turnover"].mean(), arr["turnover"].min(), arr["turnover"].max()
            )
        self._debug_buf = []
        if self.episode_days is not None:
            low  = self.window_size
            high = len(self.data) - self.episode_days - 1
            if high < low:
                start = low
            else:
                n_starts = high - low + 1
                epoch    = self._episode_idx // n_starts
                pos      = self._episode_idx % n_starts

                if pos == 0:
                    rng = np.random.default_rng(seed=epoch)
                    self._start_order = rng.permutation(n_starts)

                start = low + self._start_order[pos]

            self._episode_idx  += 1
            self.current_step   = start
            self.episode_end    = min(start + self.episode_days, len(self.data))
        else:
            self.current_step = self.window_size
            self.episode_end  = len(self.data)
        # warm-up del IPM: corre los días anteriores al inicio para que su estado sea coherente
        if self.ipm_module is not None:
            self.ipm_module.reset()
            for i in range(self.current_step - self.window_size, self.current_step):
                x = torch.tensor(self.data[i][: self.num_assets * 3], dtype=torch.float32)
                self.ipm_module.step(x)
        self.weights = np.zeros(self.num_assets + 1, dtype=np.float32)
        self.weights[-1] = 1.0
        return self._get_observation(step=self.current_step - 1), {}

    def step(self, action):
        w = self._action_to_weights(action)
        turnover = float(np.abs(w - self.weights).sum())
        total_log_ret = 0.0

        # avanza rebalance_freq días con drift natural, el agente no toca nada
        for _ in range(self.rebalance_freq):
            if self.current_step >= len(self.data):
                break
            market_data = self.data[self.current_step]
            log_ret_day = market_data[0: self.num_assets * 3: 3]
            price_rel = np.exp(log_ret_day)
            price_rel_full = np.concatenate([price_rel, [1.0]])
            portfolio_rel = np.dot(w, price_rel_full)
            total_log_ret += float(np.log(portfolio_rel + 1e-8))
            w = (w * price_rel_full) / (portfolio_rel + 1e-8)  # drift
            self.current_step += 1

        self.weights = w
        reward = self.reward_scale * total_log_ret - self.tc * turnover

        if self.debug and (len(self._debug_buf) % self.debug_every == 0):
            self._debug_buf.append({
                "port_ret": total_log_ret,
                "reward": reward,
                "turnover": turnover
            })

        terminated = self.current_step >= self.episode_end
        obs = self._get_observation(step=self.current_step - 1)
        return obs, reward, terminated, False, {"portfolio_log_ret": total_log_ret}
    
    def _action_to_weights(self, action):
        x = np.maximum(action, 0.0)   # ReLU: negativos = 0 exacto
        s = x.sum()
        if s < 1e-8:                  # si todo es 0, todo a cash
            w = np.zeros_like(x)
            w[-1] = 1.0
            return w
        return x / s

    def _get_observation(self, step=None):
        idx = self.current_step if step is None else step
        # log-returns de los últimos window_size días (señal de tendencia/momentum)
        start = idx - self.window_size + 1
        logret_window = self.data[start: idx + 1, 0: self.num_assets * 3: 3].flatten()  # (window_size * num_assets,)
        # features completas del día actual
        current_day = self.data[idx]
        pred = np.zeros(self.ipm_dim, dtype=np.float32)
        if self.ipm_module is not None:
            x_t = torch.tensor(current_day[: self.num_assets * 3], dtype=torch.float32)
            pred = self.ipm_module.step(x_t).numpy()
        obs = np.concatenate([logret_window, current_day, pred, self.weights]).astype(np.float32)
        return obs
    
    def _plot_weights(self, weights, step, every=200):
        if step % every != 0:
            return
        labels = [f"a{i+1}" for i in range(self.num_assets)] + ["cash"]
        pairs = ", ".join(f"{k}={v:.3f}" for k, v in zip(labels, weights))
        print(f"[step {step}] {pairs}")