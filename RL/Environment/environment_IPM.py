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
    def __init__(self, data, transaction_cost: float = 0.002, ipm_module=None, debug=False, debug_every=200):
        super(PortfolioEnv, self).__init__()
        self.data = data
        self.tc   = transaction_cost
        self.ipm_module = ipm_module
        self.ipm_dim = ipm_module.m if ipm_module is not None else 0
        self.debug = debug
        self.debug_every = debug_every
        self._debug_buf = []
        self.current_step = 0
        self.num_assets = 6  # 6 assets + cash
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(19 + self.num_assets + 1 + self.ipm_dim,),  # 19 + 7 = 28
            dtype=np.float32
        )
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
        if self.ipm_module is not None:
            self.ipm_module.reset()
        self.current_step = 1
        self.weights = np.zeros(self.num_assets + 1, dtype=np.float32)
        self.weights[-1] = 1.0
        return self._get_observation(step=self.current_step - 1), {}

    def step(self, action):
        w_target = self._action_to_weights(action)

        # retornos del dia t (current_step)
        market_data = self.data[self.current_step]
        log_returns = market_data[0: self.num_assets * 3: 3]
        price_rel = np.exp(log_returns)
        price_rel_full = np.concatenate([price_rel, [1.0]])

        portfolio_relative = np.dot(w_target, price_rel_full)
        port_ret = float(np.log(portfolio_relative + 1e-8))
        turnover = float(np.abs(w_target - self.weights).sum())

        reward = port_ret - self.tc * turnover

        if self.debug and (self.current_step % self.debug_every == 0):
            self._debug_buf.append({
                "port_ret": port_ret,
                "reward": reward,
                "turnover": turnover
            })

        # drift al cierre
        self.weights = (w_target * price_rel_full) / (portfolio_relative + 1e-8)

        # avanzar a dia t+1
        self.current_step += 1
        terminated = self.current_step >= len(self.data)
        obs = self._get_observation(step=self.current_step - 1)

        return obs, reward, terminated, False, {}
    
    def _action_to_weights(self, action):
        x = np.maximum(action, 0.0)   # ReLU: negativos → 0 exacto
        s = x.sum()
        if s < 1e-8:                  # si todo es 0, todo a cash
            w = np.zeros_like(x)
            w[-1] = 1.0
            return w
        return x / s

    def _get_observation(self, step=None):
        idx = self.current_step if step is None else step
        market_data = self.data[idx]
        pred = np.zeros(self.ipm_dim, dtype=np.float32)
        if self.ipm_module is not None:
            # Opcion A: usar las 18 features (sin VIX)
            x_t = torch.tensor(market_data[: self.num_assets * 3], dtype=torch.float32)
            pred = self.ipm_module.step(x_t).numpy()
        obs = np.concatenate([market_data, pred, self.weights]).astype(np.float32)
        return obs

    def _dsr(self, r: float) -> float:
        dA    = r - self._A
        dB    = r ** 2 - self._B
        var = max(self._B - self._A ** 2, 1e-4)  # o 1e-6
        denom = var ** 0.5
        dsr   = (self._B * dA - 0.5 * self._A * dB) / denom ** 3
        self._A += self.eta * dA
        self._B += self.eta * dB
        return float(dsr)

    def _calculate_reward(self, weights):
        market_data = self.data[self.current_step]

        log_returns = market_data[0: self.num_assets * 3: 3]
        price_rel = np.exp(log_returns)
        price_rel_full = np.concatenate([price_rel, [1.0]])

        portfolio_relative = np.dot(weights, price_rel_full)
        reward = np.log(portfolio_relative + 1e-8)

        # pesos despues del movimiento del mercado (drift)
        self.weights = (weights * price_rel_full) / (portfolio_relative + 1e-8)
        return reward

    def _plot_weights(self, weights, step, every=200):
        if step % every != 0:
            return
        labels = [f"a{i+1}" for i in range(self.num_assets)] + ["cash"]
        pairs = ", ".join(f"{k}={v:.3f}" for k, v in zip(labels, weights))
        print(f"[step {step}] {pairs}")