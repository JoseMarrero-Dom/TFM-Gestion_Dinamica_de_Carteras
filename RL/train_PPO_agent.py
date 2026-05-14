import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Environment.environment import PortfolioEnv
from PPO.agent import PPOAgent

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../TTS-GAN/tts-gan-tfm"))
sys.path.insert(0, ROOT)

from dataLoader import portfolio_load_dataset

def load_features(data_mode="Train", cache_dir="./data_cache"):
    dataset = portfolio_load_dataset(
        data_mode=data_mode,
        is_normalize=False,   # importante para reward con log-returns reales
        log_returns=True,
        use_intraday=True,
        use_windows=False
    )

    # (T, 18) en el orden: [a0_logret, a0_oc, a0_hl, a1_logret, ...]
    features = dataset.data[0, :, 0, :].T.astype(np.float32)

    vix_path = os.path.join(cache_dir, "portfolio_vix_20040101_20251231.csv")
    vix = pd.read_csv(vix_path, index_col=0, parse_dates=True).iloc[:, 0].values

    # Alinea longitud de VIX con T (si no tienes index exacto disponible)
    vix = vix[-features.shape[0]:].reshape(-1, 1).astype(np.float32)

    # (T, 19) = 18 features + VIX al final
    data = np.concatenate([features, vix], axis=1)
    return data

def evaluate_and_plot(model, env, freq=252, out_path=None):
    def action_to_weights(action):
        x = action - np.max(action)
        exp = np.exp(x)
        return exp / (exp.sum() + 1e-8)

    obs, _ = env.reset()
    done = False
    log_returns = []
    equity = [1.0]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        weights = action_to_weights(action)

        # log-returns en indices 0,3,6,9,12,15
        log_ret_assets = obs[0: env.num_assets * 3: 3]
        price_rel = np.exp(log_ret_assets)
        portfolio_rel = np.dot(weights[:env.num_assets], price_rel) + weights[-1]
        log_ret = np.log(portfolio_rel + 1e-8)

        log_returns.append(log_ret)
        equity.append(equity[-1] * np.exp(log_ret))

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    log_returns = np.asarray(log_returns)
    equity = np.asarray(equity)

    mean = log_returns.mean()
    std = log_returns.std() + 1e-8
    sharpe = np.sqrt(freq) * mean / std

    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    mdd = drawdown.min()

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(equity, label="Equity")
    ax[0].set_title(f"Sharpe: {sharpe:.2f}  |  MDD: {mdd:.2%}")
    ax[0].legend()
    ax[1].plot(drawdown, color="red", label="Drawdown")
    ax[1].legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()

    return sharpe, mdd

def main():
    data = load_features(data_mode="Train", cache_dir="./data_cache")

    env = PortfolioEnv(data)

    agent = PPOAgent(env)
    agent.train(total_timesteps=200_000)
    agent.save("ppo_portfolio")

    sharpe, mdd = evaluate_and_plot(agent.model, env, freq=252, out_path="eval_metrics.png")
    print("Sharpe:", sharpe, "MDD:", mdd)

if __name__ == "__main__":
    main()
