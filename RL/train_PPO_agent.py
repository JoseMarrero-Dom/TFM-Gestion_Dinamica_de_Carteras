import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch


from Environment.environment import PortfolioEnv
from Environment.environment_IPM import PortfolioEnv as PortfolioEnvIPM
from PPO.agent import PPOAgent
from IPM.ipm import IPMModule

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../TTS-GAN/tts-gan-tfm"))
sys.path.insert(0, ROOT)

from dataLoader import portfolio_load_dataset

def load_features(data_mode="Train", cache_dir="./data_cache"):
    dataset = portfolio_load_dataset(
        data_mode=data_mode,
        is_normalize=False,
        log_returns=True,
        use_intraday=True,
        use_windows=False
    )

    features = dataset.data[0, :, 0, :].T.astype(np.float32)

    vix_path = os.path.join(cache_dir, "portfolio_vix_20040101_20251231.csv")
    vix = pd.read_csv(vix_path, index_col=0, parse_dates=True).iloc[:, 0].values
    vix = vix[-features.shape[0]:].reshape(-1, 1).astype(np.float32)

    return np.concatenate([features, vix], axis=1)

def evaluate_and_plot(model, env, freq=52, var_conf=0.95, out_path=None):
    def action_to_weights(action):
        x = np.maximum(action, 0.0)
        s = x.sum()
        if s < 1e-8:
            w = np.zeros_like(x)
            w[-1] = 1.0
            return w
        return x / s

    asset_names = ["SP500", "MSCI_EAFE", "MSCI_EM", "Gold", "Oil_WTI", "UST10Y", "Cash"]

    obs, _ = env.reset()
    done = False
    log_returns  = []
    equity       = [1.0]
    weights_hist = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        weights_hist.append(action_to_weights(action).copy())

        obs, reward, terminated, truncated, _ = env.step(action)
        log_returns.append(float(reward))
        equity.append(equity[-1] * np.exp(float(reward)))
        done = terminated or truncated

    log_returns  = np.asarray(log_returns)
    equity       = np.asarray(equity)
    weights_hist = np.asarray(weights_hist)

    mean = log_returns.mean()
    std  = log_returns.std() + 1e-8
    sharpe  = np.sqrt(freq) * mean / std

    downside = log_returns[log_returns < 0]
    semi_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 1e-8
    sortino  = np.sqrt(freq) * mean / semi_std

    peak     = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    mdd      = drawdown.min()
    var      = np.percentile(log_returns, (1 - var_conf) * 100)

    title = (f"Sharpe: {sharpe:.2f}  |  Sortino: {sortino:.2f}  |  "
             f"MDD: {mdd:.2%}  |  VaR({int(var_conf*100)}%): {var:.4f}")
    print(title)

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=False)
    axes[0].plot(equity)
    axes[0].set_title(title, fontsize=9)
    axes[0].set_ylabel("Equity")
    axes[1].plot(drawdown, color="red")
    axes[1].axhline(mdd, color="darkred", linestyle="--", linewidth=0.8, label=f"MDD {mdd:.2%}")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[2].hist(log_returns, bins=60, color="steelblue", edgecolor="none")
    axes[2].axvline(var,  color="red",   linestyle="--", linewidth=1.2, label=f"VaR {var:.4f}")
    axes[2].axvline(mean, color="green", linestyle="--", linewidth=1.2, label=f"Media {mean:.4f}")
    axes[2].set_ylabel("Frecuencia")
    axes[2].set_xlabel("Log-return semanal")
    axes[2].legend()
    axes[3].boxplot(weights_hist, labels=asset_names, patch_artist=True)
    axes[3].set_ylabel("Peso cartera")
    margin = 0.05
    axes[3].set_ylim(max(0, weights_hist.min() - margin), min(1, weights_hist.max() + margin))
    axes[3].tick_params(axis="x", labelrotation=20)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()

    n = len(asset_names)
    fig2, axes2 = plt.subplots(n, 1, figsize=(10, 2 * n), sharex=True)
    for i, name in enumerate(asset_names):
        axes2[i].plot(weights_hist[:, i], linewidth=0.8)
        axes2[i].set_ylabel(name, fontsize=8)
        axes2[i].set_ylim(0, 1)
    axes2[-1].set_xlabel("Semana")
    fig2.suptitle("Peso por activo (semana a semana)", fontsize=10)
    plt.tight_layout()
    if out_path:
        fig2.savefig(out_path.replace(".", "_weights."), dpi=150)
    else:
        plt.show()

    return {"sharpe": sharpe, "sortino": sortino, "mdd": mdd, "var": var}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    data      = load_features(data_mode="Train", cache_dir="./data_cache")
    data_test = load_features(data_mode="Test",  cache_dir="./data_cache")
    set_seed(0)

    ipm = IPMModule(m=18)
    env = PortfolioEnvIPM(data, ipm_module=ipm, debug=False, episode_weeks=52)

    agent = PPOAgent(env, seed=0)
    agent.train(total_timesteps=500_000)

    agent.save("ppo_portfolio")

    env_final  = PortfolioEnvIPM(data_test, ipm_module=ipm, debug=False, episode_weeks=None)
    evaluate_and_plot(agent.model, env_final, out_path="eval_metrics.png")

if __name__ == "__main__":
    main()
