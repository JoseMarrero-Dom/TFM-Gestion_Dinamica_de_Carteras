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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── paths ─────────────────────────────────────────────────────────────────────
RL_DIR   = os.path.dirname(__file__)
GAN_DIR  = os.path.abspath(os.path.join(RL_DIR, "../TTS-GAN/tts-gan-tfm"))
LOGS_DIR = os.path.join(GAN_DIR, "logs")
sys.path.insert(0, GAN_DIR)

from GANModels import Generator
from dataLoader import portfolio_load_dataset, DEFAULT_TICKERS

from Environment.environment_IPM import PortfolioEnv
from PPO.agent import PPOAgent
from IPM.ipm import IPMModule

# ── constantes ────────────────────────────────────────────────────────────────
ALL_ASSETS  = list(DEFAULT_TICKERS.keys())   # orden fijo del dataLoader
N_ASSETS    = len(ALL_ASSETS)                # 6
VIX_STRESS  = 25.0   # valor VIX representativo para las ventanas sintéticas
N_SYNTHETIC = 2000     # ventanas sintéticas a generar por modelo


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
    if cache_dir is None:
        cache_dir = os.path.join(RL_DIR, "data_cache")
    ds = portfolio_load_dataset(
        data_mode=data_mode,
        split_date="2021-01-01",
        is_normalize=False,
        log_returns=True,
        use_intraday=True,
        use_windows=False,
    )
    features = ds.data[0, :, 0, :].T.astype(np.float32)  # (T, 18)
    T = features.shape[0]

    vix_path = os.path.join(cache_dir, "portfolio_vix_20040101_20251231.csv")
    vix_full = pd.read_csv(vix_path, index_col=0, parse_dates=True).iloc[:, 0]

    if data_mode == "Train":
        vix = vix_full.values[:T]
    else:
        vix = vix_full.values[-T:]

    vix = vix.reshape(-1, 1).astype(np.float32)
    return np.concatenate([features, vix], axis=1)  # (T, 19)


def load_stress_windows(seq_len, n_samples=1000):
    """Ventanas reales de régimen moderate+stress — shape (N, T, 18)."""
    ds = portfolio_load_dataset(
        data_mode="Train",
        split_date="2021-01-01",
        is_normalize=False,
        log_returns=True,
        use_intraday=True,
        use_windows=True,
        window_length=seq_len,
        stride=8,
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


def denormalize(windows_norm, mean, std, clip_sigma=4.0):
    """Convierte ventanas z-score → escala de retornos reales.
    Recorta a ±clip_sigma desviaciones para evitar overflow en np.exp().
    """
    out = windows_norm * std + mean
    lo  = mean - clip_sigma * std
    hi  = mean + clip_sigma * std
    return np.clip(out, lo, hi)


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


def windows_to_flat(windows_denorm, vix_value=VIX_STRESS, log_ret_clip=0.30):
    """
    Convierte ventanas (N, T, C) en array plano (N*T, C+1) añadiendo VIX.
    Se usa para modelos multi-activo (C=18).
    Recorta log-returns a ±log_ret_clip para evitar overflow en np.exp().
    """
    N, T, C = windows_denorm.shape
    flat = np.clip(windows_denorm.reshape(-1, C), -log_ret_clip, log_ret_clip)
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
from typing import Callable

def linear_schedule(initial_value: float, floor_value: float = 1e-5) -> Callable[[float], float]:
    """
    Planificador de tasa de aprendizaje lineal.
    
    Args:
        initial_value: El learning rate con el que empezará el entrenamiento.
        floor_value: El learning rate mínimo permitido.
    Returns:
        Una función que calcula el learning rate actual basado en el progreso restante.
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining va desde 1.0 (inicio) hasta 0.0 (final del entrenamiento)
        lr = progress_remaining * initial_value
        return max(lr, floor_value)
    return func

def train_and_eval(real_data, test_data, ipm, out_dir,
                   synthetic_data=None,
                   timesteps_real=2_500_000,
                   timesteps_finetune=20_000_000):
    """
    Fase 1: entrena PPO solo con datos reales.
    Fase 2: si hay sintéticos, continúa entrenando con reales + sintéticos
            partiendo del modelo ya aprendido (curriculum learning).
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Fase 1: solo real ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Fase 1 — entrenando PPO con datos reales ({len(real_data)} pasos)")
    env_real  = PortfolioEnv(real_data, ipm_module=ipm, episode_weeks=52, reward_scale=50)
    env_wrapped = DummyVecEnv([lambda: env_real])
    env_normalized = VecNormalize(env_wrapped, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    agent     = PPOAgent(env_normalized, seed=0, learning_rate=3e-4)
    agent.train(total_timesteps=timesteps_real)
    agent.save(os.path.join(out_dir, "ppo_baseline"))
    # Guardamos las estadísticas de normalización de la Fase 1
    env_normalized.save(os.path.join(out_dir, "vec_normalize_baseline.pkl"))

    # EVALUACIÓN FASE 1: Creamos el entorno de test normalizado con datos de Fase 1
    raw_env_test = PortfolioEnv(test_data, ipm_module=ipm, episode_weeks=None)
    env_test_vec = DummyVecEnv([lambda: raw_env_test])
    env_test_normalized = VecNormalize(env_test_vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Copiamos y congelamos estadísticas para el test
    env_test_normalized.obs_rms = env_normalized.obs_rms
    env_test_normalized.training = False
    env_test_normalized.norm_reward = False

    metrics_base = evaluate_and_plot(
        agent.model, env_test_normalized, # <── Pasamos el entorno vectorizado y normalizado
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

    env_aug = PortfolioEnv(augmented, ipm_module=ipm, episode_weeks=52, reward_scale=50)
    env_wrapped_aug = DummyVecEnv([lambda: env_aug])
    env_normalized_aug = VecNormalize(env_wrapped_aug, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # CRÍTICO: Transferir la experiencia de normalización de la Fase 1 a la Fase 2
    env_normalized_aug.obs_rms = env_normalized.obs_rms
    
    agent.model.set_env(env_normalized_aug)
    agent.model.lr_schedule = linear_schedule(3e-4)
    agent.train(total_timesteps=timesteps_finetune)
    agent.save(os.path.join(out_dir, "ppo_augmented"))
    env_normalized_aug.save(os.path.join(out_dir, "vec_normalize_augmented.pkl"))

    # EVALUACIÓN FASE 2: Sincronizamos el entorno de test con las nuevas estadísticas
    env_test_normalized_aug = VecNormalize(env_test_vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
    env_test_normalized_aug.obs_rms = env_normalized_aug.obs_rms
    env_test_normalized_aug.training = False
    env_test_normalized_aug.norm_reward = False

    metrics_aug = evaluate_and_plot(
        agent.model, env_test_normalized_aug, # <── Pasamos el nuevo entorno de test adaptado
        out_path=os.path.join(out_dir, "eval_augmented.png")
    )
    print(f"Aumentado → Sharpe: {metrics_aug['sharpe']:.3f}  "
          f"Sortino: {metrics_aug['sortino']:.3f}  "
          f"MDD: {metrics_aug['mdd']:.2%}  VaR: {metrics_aug['var']:.4f}")

    return {"baseline": metrics_base, "augmented": metrics_aug}

def evaluate_and_plot(model, env, freq=52, var_conf=0.95, out_path=None):
    asset_names = ALL_ASSETS + ["Cash"]

    obs = env.reset()
    done   = False
    log_returns, equity, weights_hist = [], [1.0], []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        w = _action_to_weights(action)
        weights_hist.append(w.copy())
        obs, reward, dones, infos = env.step(action)
        info = infos[0]
        port_ret = float(info["portfolio_log_ret"])
        log_returns.append(port_ret)
        equity.append(equity[-1] * np.exp(port_ret))
        done = bool(dones[0])

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
        and not d.startswith("portfolio")
    )
    def _find_ckpt(run):
        for name in ("checkpoint", "checkpoint.zip"):
            p = os.path.join(LOGS_DIR, run, "Model", name)
            if os.path.exists(p):
                return p
        return None

    checkpoints = [(run, _find_ckpt(run)) for run in runs if _find_ckpt(run)]

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
            market_flat = np.clip(augmented_windows.reshape(-1, 18), -0.30, 0.30)
            synth_flat = np.concatenate([market_flat, vix_col], axis=1)

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
        ipm=IPMModule(m=18),
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
