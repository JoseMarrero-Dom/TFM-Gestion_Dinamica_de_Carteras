"""Comparison plot for the equity evolution of portfolio models.

The script overlays the equity curves of the four strategies used in the
project:
  - PPO baseline
  - PPO augmented
  - static buy & hold
  - 90/10 Buffett allocation

The PPO strategies are evaluated on their own rebalancing schedule and the
static strategies are sampled on the same dates so the chart is comparable.
"""

import argparse
import importlib.util
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../RL"))
GAN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../TTS-GAN/tts-gan-tfm"))
sys.path.insert(0, RL_DIR)
sys.path.insert(0, GAN_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {module_name} desde {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data_loader = load_module("dataLoader", os.path.join(GAN_DIR, "dataLoader.py"))
environment_ipm = load_module("environment_IPM", os.path.join(RL_DIR, "Environment", "environment_IPM.py"))
ipm_module = load_module("ipm", os.path.join(RL_DIR, "IPM", "ipm.py"))

DEFAULT_TICKERS = data_loader.DEFAULT_TICKERS
portfolio_load_dataset = data_loader.portfolio_load_dataset
PortfolioEnvIPM = environment_ipm.PortfolioEnv
IPMModule = ipm_module.IPMModule


SPLIT_DATE = "2021-01-01"
ASSET_NAMES = list(DEFAULT_TICKERS.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="Compare the equity evolution of portfolio models.")
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default=os.path.join(RL_DIR, "results_baseline"),
        help="Directory with the baseline PPO model.",
    )
    parser.add_argument(
        "--augmented_dir",
        type=str,
        default=os.path.join(RL_DIR, "results_gan_augmented"),
        help="Directory with the augmented PPO model.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.join(RL_DIR, "data_cache"),
        help="Directory with cached portfolio CSV files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results_comparison"),
        help="Directory where the combined plot and CSV will be saved.",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="comparison_equity.png",
        help="Filename for the combined equity figure.",
    )
    return parser.parse_args()


def list_models(model_dir):
    return sorted(f[:-4] for f in os.listdir(model_dir) if f.endswith(".zip"))


def find_model(model_dir, model_name=None):
    available = list_models(model_dir)
    if not available:
        raise FileNotFoundError(f"No hay modelos .zip en {model_dir}")
    if model_name:
        if model_name not in available:
            raise FileNotFoundError(f"'{model_name}' no encontrado en {model_dir}. Disponibles: {available}")
        chosen = model_name
    else:
        chosen = sorted(
            available,
            key=lambda name: os.path.getmtime(os.path.join(model_dir, name + ".zip")),
            reverse=True,
        )[0]
    path = os.path.join(model_dir, chosen + ".zip")
    print(f"Modelo cargado: {path}")
    return PPO.load(path, device="cpu"), chosen


def find_vec_normalize(model_dir, model_name):
    suffix = model_name.removeprefix("ppo_")
    candidates = [
        os.path.join(model_dir, f"vec_normalize_{suffix}.pkl"),
        os.path.join(model_dir, f"vec_normalize_{model_name}.pkl"),
        os.path.join(model_dir, "vec_normalize.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"VecNormalize cargado: {path}")
            return path
    raise FileNotFoundError(f"No se encontro VecNormalize en {model_dir}. Probados: {candidates}")


def load_test_data(cache_dir):
    ds = portfolio_load_dataset(
        data_mode="Test",
        split_date=SPLIT_DATE,
        is_normalize=False,
        log_returns=True,
        use_intraday=True,
        use_windows=False,
        cache_dir=cache_dir,
    )
    features = ds.data[0, :, 0, :].T.astype(np.float32)

    prices_csv = os.path.join(cache_dir, "portfolio_prices_20040101_20251231.csv")
    if os.path.exists(prices_csv):
        all_dates = pd.read_csv(prices_csv, index_col=0, parse_dates=True).index
        test_dates = all_dates[all_dates >= pd.Timestamp(SPLIT_DATE)][1:]
        dates = pd.DatetimeIndex(test_dates[: len(features)])
    else:
        dates = pd.date_range(start=SPLIT_DATE, periods=len(features), freq="B")

    vix_path = os.path.join(cache_dir, "portfolio_vix_20040101_20251231.csv")
    vix_full = pd.read_csv(vix_path, index_col=0, parse_dates=True).iloc[:, 0]
    vix_test = vix_full.reindex(dates, method="ffill").fillna(20.0).values

    n = min(len(features), len(vix_test), len(dates))
    features, vix_test, dates = features[:n], vix_test[:n], dates[:n]

    data = np.concatenate([features, vix_test.reshape(-1, 1).astype(np.float32)], axis=1)
    return data, dates


def simulate_buy_and_hold(asset_log_returns, initial_weights):
    asset_values = initial_weights[None, :] * np.exp(np.cumsum(asset_log_returns, axis=0))
    portfolio_wealth = asset_values.sum(axis=1)
    log_returns = np.diff(np.log(np.concatenate([[1.0], portfolio_wealth])))
    equity = np.exp(np.cumsum(log_returns))
    return log_returns.astype(np.float32), equity.astype(np.float32)


def run_ppo(model, data, vecnormalize_path):
    ipm = IPMModule(m=18)
    raw_env = PortfolioEnvIPM(data, ipm_module=ipm, episode_weeks=None, rebalance_freq=5)
    rebalance_freq = raw_env.rebalance_freq

    env = DummyVecEnv([lambda: raw_env])
    env = VecNormalize.load(vecnormalize_path, env)
    env.training = False
    env.norm_reward = False

    obs = env.reset()
    done = False
    log_returns = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        step_outputs = env.step(action)
        if len(step_outputs) == 4:
            obs, reward, dones, infos = step_outputs
            info = infos[0]
            done = bool(dones)
        else:
            obs, reward, terminated, truncated, info = step_outputs
            done = terminated or truncated

        port_ret = float(info.get("portfolio_log_ret", reward))
        log_returns.append(port_ret)

    decision_idx = np.arange(rebalance_freq, rebalance_freq + len(log_returns) * rebalance_freq, rebalance_freq)
    decision_idx = decision_idx[decision_idx < len(data)]
    equity = np.exp(np.cumsum(np.asarray(log_returns, dtype=np.float32)))
    equity = equity[: len(decision_idx)]
    return decision_idx, equity


def sample_equity(series_dates, equity, sample_dates):
    index = pd.DatetimeIndex(series_dates)
    if len(equity) != len(index):
        raise ValueError(f"Length mismatch: equity={len(equity)} dates={len(index)}")

    series = pd.Series(equity, index=index)
    sampled = series.reindex(pd.DatetimeIndex(sample_dates), method="ffill")
    return sampled.index, sampled.to_numpy(dtype=np.float64)


def build_comparison_series(dates, data, baseline_dir, augmented_dir):
    baseline_model, baseline_name = find_model(baseline_dir)
    baseline_vec = find_vec_normalize(baseline_dir, baseline_name)
    baseline_idx, baseline_eq = run_ppo(baseline_model, data, baseline_vec)
    baseline_dates = dates[baseline_idx]

    augmented_model, augmented_name = find_model(augmented_dir)
    augmented_vec = find_vec_normalize(augmented_dir, augmented_name)
    augmented_idx, augmented_eq = run_ppo(augmented_model, data, augmented_vec)
    augmented_dates = dates[augmented_idx]

    common_dates = pd.DatetimeIndex(baseline_dates)
    common_dates = common_dates.intersection(pd.DatetimeIndex(augmented_dates))
    sample_dates = common_dates

    features = data[:, :18]
    asset_log_returns = features[:, ::3]

    static_weights = np.full(len(ASSET_NAMES), 1.0 / len(ASSET_NAMES), dtype=np.float32)
    _, static_eq = simulate_buy_and_hold(asset_log_returns, static_weights)
    static_dates = dates[: len(static_eq)]
    static_dates, static_eq = sample_equity(static_dates, static_eq, sample_dates)

    buffalo_weights = np.array([0.9] + [0.0] * (len(ASSET_NAMES) - 2) + [0.1], dtype=np.float32)
    _, buffalo_eq = simulate_buy_and_hold(asset_log_returns, buffalo_weights)
    buffalo_dates = dates[: len(buffalo_eq)]
    buffalo_dates, buffalo_eq = sample_equity(buffalo_dates, buffalo_eq, sample_dates)

    baseline_dates, baseline_eq = sample_equity(baseline_dates, baseline_eq, sample_dates)
    augmented_dates, augmented_eq = sample_equity(augmented_dates, augmented_eq, sample_dates)

    comparison = pd.DataFrame(
        {
            "baseline": baseline_eq,
            "augmented": augmented_eq,
            "static": static_eq,
            "90_10": buffalo_eq,
        },
        index=sample_dates,
    )
    comparison.index.name = "date"
    return comparison


def plot_comparison(comparison, out_path):
    fig, ax = plt.subplots(figsize=(15, 7))

    styles = {
        "baseline": {"color": "#1f77b4", "linestyle": "-"},
        "augmented": {"color": "#d62728", "linestyle": "-"},
        "static": {"color": "#2ca02c", "linestyle": "--"},
        "90_10": {"color": "#9467bd", "linestyle": "-."},
    }
    labels = {
        "baseline": "Baseline PPO",
        "augmented": "Augmented PPO",
        "static": "Static buy & hold",
        "90_10": "90/10 Buffett",
    }

    for column in comparison.columns:
        ax.plot(
            comparison.index,
            comparison[column],
            label=labels[column],
            linewidth=2.0,
            **styles[column],
        )

    ax.set_title("Evolucion del equity de las estrategias de cartera (2021-2025)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Equity (base 1.0)")
    ax.set_xlabel("Fecha")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", fontsize=16, ncol=2)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado -> {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Cargando datos 2021+...")
    data, dates = load_test_data(args.cache_dir)
    print(f"  Periodo: {dates[0].date()} -> {dates[-1].date()} ({len(dates)} dias de trading)")

    comparison = build_comparison_series(dates, data, args.baseline_dir, args.augmented_dir)

    csv_path = os.path.join(args.out_dir, "comparison_equity.csv")
    comparison.to_csv(csv_path, float_format="%.6f")
    print(f"CSV comparativo -> {csv_path}")

    plot_comparison(comparison, os.path.join(args.out_dir, args.out_name))


if __name__ == "__main__":
    main()