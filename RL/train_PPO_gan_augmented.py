"""
Entrena PPO con datos reales aumentados con eventos de cola sintéticos del TTS-GAN.

Flujo:
  1. Carga los checkpoints disponibles en TTS-GAN/logs/
  2. Por cada modelo genera N ventanas sintéticas (moderate+stress)
  3. Desnormaliza las ventanas al espacio de retornos reales
  4. Construye el array de entrenamiento: [real | sintético_modelo1 | ...]
  5. Entrena PPO y evalúa en datos de test reales
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
RL_DIR   = os.path.dirname(__file__)
GAN_DIR  = os.path.abspath(os.path.join(RL_DIR, "../TTS-GAN/tts-gan-tfm"))
LOGS_DIR = os.path.join(GAN_DIR, "logs")
sys.path.insert(0, GAN_DIR)

from GANModels import Generator
from dataLoader import portfolio_load_dataset, DEFAULT_TICKERS

from Environment.environment import PortfolioEnv
from PPO.agent import PPOAgent

# ── constantes ────────────────────────────────────────────────────────────────
ALL_ASSETS  = list(DEFAULT_TICKERS.keys())   # orden fijo del dataLoader
N_ASSETS    = len(ALL_ASSETS)                # 6
VIX_STRESS  = 25.0   # valor VIX representativo para las ventanas sintéticas
N_SYNTHETIC = 500     # ventanas sintéticas a generar por modelo


# ── utilidades GAN ────────────────────────────────────────────────────────────

def _infer_gen_config(state_dict):
    channels   = state_dict["deconv.0.weight"].shape[0]
    latent_dim = state_dict["l1.weight"].shape[1]
    seq_len    = state_dict["pos_embed"].shape[1]
    return channels, latent_dim, seq_len


def _detect_asset(run_name, channels):
    for asset in ALL_ASSETS:
        if asset.upper() in run_name.upper():
            return asset
    return "UST10Y" if channels <= 3 else None


def _load_generator(ckpt_path):
    ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd         = ckpt["avg_gen_state_dict"]
    channels, latent_dim, seq_len = _infer_gen_config(sd)
    gen = Generator(seq_len=seq_len, patch_size=15, channels=channels, latent_dim=latent_dim)
    gen.load_state_dict(sd)
    gen.eval()
    return gen, channels, latent_dim, seq_len


def _generate_windows(gen, latent_dim, n):
    """Devuelve (n, seq_len, channels) en espacio normalizado z-score."""
    z = torch.randn(n, latent_dim)
    with torch.no_grad():
        out = gen(z).cpu().numpy()          # (n, C, 1, T)
    return np.transpose(out.squeeze(2), (0, 2, 1))  # (n, T, C)


# ── carga de datos reales ──────────────────────────────────────────────────────

def load_real_flat(data_mode="Train", cache_dir=None):
    """Carga datos reales sin normalizar ni ventanear — shape (T, 19)."""
    if cache_dir is None:
        cache_dir = os.path.join(RL_DIR, "data_cache")
    ds = portfolio_load_dataset(
        data_mode=data_mode,
        is_normalize=False,
        log_returns=True,
        use_intraday=True,
        use_windows=False,
    )
    features = ds.data[0, :, 0, :].T.astype(np.float32)   # (T, 18)

    vix_path = os.path.join(cache_dir, "portfolio_vix_20040101_20251231.csv")
    vix = pd.read_csv(vix_path, index_col=0, parse_dates=True).iloc[:, 0].values
    vix = vix[-features.shape[0]:].reshape(-1, 1).astype(np.float32)

    return np.concatenate([features, vix], axis=1)  # (T, 19)


def load_stress_windows(seq_len, n_samples=1000):
    """Ventanas reales de régimen moderate+stress — shape (N, T, 18)."""
    ds = portfolio_load_dataset(
        data_mode="Train",
        is_normalize=False,
        log_returns=True,
        use_intraday=True,
        use_windows=True,
        window_length=seq_len,
        stride=1,
        label_mode="regime",
        filter_regime=["moderate", "stress"],
    )
    N = min(n_samples, len(ds))
    windows = np.stack([ds[i][0] for i in range(N)])   # (N, 18, 1, T)
    return np.transpose(windows.squeeze(2), (0, 2, 1))  # (N, T, 18)


# ── desnormalización ──────────────────────────────────────────────────────────

def compute_channel_stats(real_flat):
    """Media y std global de cada canal (sobre los 18 canales de mercado)."""
    market = real_flat[:, :18]   # excluir VIX
    return market.mean(axis=0), market.std(axis=0) + 1e-8


def denormalize(windows_norm, mean, std):
    """Convierte ventanas z-score → escala de retornos reales."""
    return windows_norm * std + mean


# ── construcción del dataset aumentado ────────────────────────────────────────

def build_augmented_data(real_flat, synthetic_blocks):
    """
    Concatena bloques sintéticos al final del array real.

    Args:
        real_flat:        (T, 19)
        synthetic_blocks: lista de arrays (T_i, 19)

    Returns:
        (T + sum(T_i), 19)
    """
    parts = [real_flat] + synthetic_blocks
    return np.concatenate(parts, axis=0).astype(np.float32)


def windows_to_flat(windows_denorm, vix_value=VIX_STRESS):
    """
    Convierte ventanas (N, T, C) en array plano (N*T, C+1) añadiendo VIX.
    Se usa para modelos multi-activo (C=18).
    """
    N, T, C = windows_denorm.shape
    flat = windows_denorm.reshape(-1, C)
    vix_col = np.full((N * T, 1), vix_value, dtype=np.float32)
    return np.concatenate([flat, vix_col], axis=1)


def inject_single_asset(stress_windows_real, gen_windows_norm, asset_name, mean, std):
    """
    Para modelos de un activo: sustituye los canales de ese activo
    en ventanas reales de stress por los canales generados por el GAN.

    Args:
        stress_windows_real: (N, T, 18) retornos reales en ventanas stress
        gen_windows_norm:    (N, T, 3)  salida normalizada del GAN
        asset_name:          nombre del activo, ej. "UST10Y"
        mean, std:           (18,) stats globales de todos los canales

    Returns:
        (N, T, 18) con los canales del activo sustituidos
    """
    asset_idx   = ALL_ASSETS.index(asset_name)
    col_start   = asset_idx * 3
    col_end     = col_start + 3

    asset_mean  = mean[col_start:col_end]
    asset_std   = std[col_start:col_end]
    gen_denorm  = denormalize(gen_windows_norm, asset_mean, asset_std)

    out = stress_windows_real.copy()
    N   = min(len(out), len(gen_denorm))
    out[:N, :, col_start:col_end] = gen_denorm[:N]
    return out[:N]


# ── entrenamiento PPO ──────────────────────────────────────────────────────────

def train_and_eval(real_data, test_data, out_dir,
                   synthetic_data=None,
                   timesteps_real=300_000,
                   timesteps_finetune=150_000):
    """
    Fase 1: entrena PPO solo con datos reales.
    Fase 2: si hay sintéticos, continúa entrenando con reales + sintéticos
            partiendo del modelo ya aprendido (curriculum learning).
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Fase 1: solo real ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Fase 1 — entrenando PPO con datos reales ({len(real_data)} pasos)")
    env_real  = PortfolioEnv(real_data)
    agent     = PPOAgent(env_real, seed=42)
    agent.train(total_timesteps=timesteps_real)
    agent.save(os.path.join(out_dir, "ppo_baseline"))

    env_test = PortfolioEnv(test_data)
    metrics_base = evaluate_and_plot(
        agent.model, env_test,
        out_path=os.path.join(out_dir, "eval_baseline.png")
    )
    print(f"Baseline  → Sharpe: {metrics_base['sharpe']:.3f}  "
          f"Sortino: {metrics_base['sortino']:.3f}  "
          f"MDD: {metrics_base['mdd']:.2%}  VaR: {metrics_base['var']:.4f}")

    if synthetic_data is None:
        return {"baseline": metrics_base}

    # ── Fase 2: fine-tune con real + sintético ─────────────────────────
    augmented = build_augmented_data(real_data, [synthetic_data])
    n_synth   = len(augmented) - len(real_data)
    print(f"\nFase 2 — fine-tune con reales + sintéticos")
    print(f"  {len(real_data)} reales + {n_synth} sintéticos = {len(augmented)} total")

    env_aug = PortfolioEnv(augmented)
    agent.model.set_env(env_aug)
    agent.train(total_timesteps=timesteps_finetune)
    agent.save(os.path.join(out_dir, "ppo_augmented"))

    metrics_aug = evaluate_and_plot(
        agent.model, env_test,
        out_path=os.path.join(out_dir, "eval_augmented.png")
    )
    print(f"Aumentado → Sharpe: {metrics_aug['sharpe']:.3f}  "
          f"Sortino: {metrics_aug['sortino']:.3f}  "
          f"MDD: {metrics_aug['mdd']:.2%}  VaR: {metrics_aug['var']:.4f}")

    return {"baseline": metrics_base, "augmented": metrics_aug}


def evaluate_and_plot(model, env, freq=52, var_conf=0.95, out_path=None):
    asset_names = ALL_ASSETS + ["Cash"]

    obs, _ = env.reset()
    done   = False
    log_returns, equity, weights_hist = [], [1.0], []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        w = _action_to_weights(action)
        weights_hist.append(w.copy())
        obs, reward, terminated, truncated, _ = env.step(action)
        # reconstruir log-ret del portfolio desde reward (reward = log_ret - tc*turnover)
        log_returns.append(float(reward))
        equity.append(equity[-1] * np.exp(float(reward)))
        done = terminated or truncated

    lr  = np.asarray(log_returns)
    eq  = np.asarray(equity)
    wh  = np.asarray(weights_hist)

    sharpe  = np.sqrt(freq) * lr.mean() / (lr.std() + 1e-8)
    down    = lr[lr < 0]
    sortino = np.sqrt(freq) * lr.mean() / (np.sqrt((down**2).mean()) + 1e-8) if len(down) else float("inf")
    peak    = np.maximum.accumulate(eq)
    mdd     = (eq / peak - 1).min()
    var     = np.percentile(lr, (1 - var_conf) * 100)

    title = f"Sharpe {sharpe:.2f} | Sortino {sortino:.2f} | MDD {mdd:.2%} | VaR {var:.4f}"
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    axes[0].plot(eq);          axes[0].set_title(title, fontsize=9); axes[0].set_ylabel("Equity")
    axes[1].plot(eq/peak - 1, color="red"); axes[1].set_ylabel("Drawdown")
    axes[2].hist(lr, bins=60, color="steelblue"); axes[2].set_ylabel("Frecuencia")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.close(fig)

    return {"sharpe": sharpe, "sortino": sortino, "mdd": mdd, "var": var}


def _action_to_weights(action):
    x = np.maximum(action, 0.0)
    s = x.sum()
    if s < 1e-8:
        w = np.zeros_like(x); w[-1] = 1.0; return w
    return x / s


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.join(RL_DIR, "results_gan_augmented")
    os.makedirs(out_dir, exist_ok=True)

    print("Cargando datos reales...")
    real_train = load_real_flat(data_mode="Train")
    real_test  = load_real_flat(data_mode="Test")
    mean, std  = compute_channel_stats(real_train)

    # Buscar checkpoints
    runs = sorted(
        d for d in os.listdir(LOGS_DIR)
        if os.path.isdir(os.path.join(LOGS_DIR, d))
    )
    checkpoints = [
        (run, os.path.join(LOGS_DIR, run, "Model", "checkpoint"))
        for run in runs
        if os.path.exists(os.path.join(LOGS_DIR, run, "Model", "checkpoint"))
    ]

    # Recoger bloques sintéticos de todos los modelos
    synthetic_blocks = []
    stress_real = None

    for run_name, ckpt_path in checkpoints:
        print(f"\nProcesando checkpoint: {run_name}")
        gen, channels, latent_dim, seq_len = _load_generator(ckpt_path)
        asset = _detect_asset(run_name, channels)

        gen_windows = _generate_windows(gen, latent_dim, N_SYNTHETIC)

        if channels == 18:
            synth_denorm = denormalize(gen_windows, mean, std)
            synth_flat   = windows_to_flat(synth_denorm)

        elif channels in (1, 3):
            if stress_real is None:
                print("Cargando ventanas reales de stress para inyección...")
                stress_real = load_stress_windows(seq_len, n_samples=N_SYNTHETIC)

            n_use = min(len(stress_real), len(gen_windows))
            augmented_windows = inject_single_asset(
                stress_real[:n_use], gen_windows[:n_use], asset, mean, std
            )
            vix_col    = np.full((n_use * seq_len, 1), VIX_STRESS, dtype=np.float32)
            synth_flat = np.concatenate([augmented_windows.reshape(-1, 18), vix_col], axis=1)

        else:
            print(f"  channels={channels} no soportado, saltando.")
            continue

        print(f"  {run_name}: {len(synth_flat)} pasos añadidos al pool")
        synthetic_blocks.append(synth_flat)

    # Un solo bloque sintético combinado de todos los modelos
    all_synthetic = np.concatenate(synthetic_blocks, axis=0) if synthetic_blocks else None

    # Fase 1 (solo real) + Fase 2 (fine-tune con real+sintético)
    results = train_and_eval(
        real_data=real_train,
        test_data=real_test,
        out_dir=out_dir,
        synthetic_data=all_synthetic,
    )

    # Resumen comparativo
    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")
    print(f"{'Fase':<20} {'Sharpe':>7} {'Sortino':>8} {'MDD':>8} {'VaR':>8}")
    print("-" * 55)
    for fase, m in results.items():
        print(f"{fase:<20} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['mdd']:>7.2%} {m['var']:>8.4f}")

    import csv
    csv_path = os.path.join(out_dir, "comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fase", "sharpe", "sortino", "mdd", "var"])
        w.writeheader()
        for fase, m in results.items():
            w.writerow({"fase": fase, **m})
    print(f"\nTabla guardada → {csv_path}")


if __name__ == "__main__":
    main()
