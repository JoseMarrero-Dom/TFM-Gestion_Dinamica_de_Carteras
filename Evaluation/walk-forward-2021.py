"""
Walk-forward backtesting 2021–2025 del agente PPO.

Métricas: Sharpe (1994), Sortino (1994), MDD — desagregadas por subperiodo
de régimen VIX y por episodios explícitos (guerra Rusia-Ucrania 2022,
crisis bancaria regional 2023).

Test de Diebold-Mariano (1995) entre el modelo principal y un baseline
opcional para determinar significación estadística de la diferencia.

Uso:
    # listar modelos disponibles
    python walk-forward-2021.py --model_dir ../RL/results_gan_augmented --list

    # evaluar un modelo concreto
    python walk-forward-2021.py --model_dir ../RL/results_gan_augmented \\
                                --model_name ppo_augmented

    # evaluar + comparar con baseline (test DM)
    python walk-forward-2021.py --model_dir ../RL/results_gan_augmented \\
                                --model_name ppo_augmented \\
                                --baseline_dir ../RL \\
                                --baseline_name ppo_portfolio
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

RL_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../RL"))
GAN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../TTS-GAN/tts-gan-tfm"))
sys.path.insert(0, RL_DIR)
sys.path.insert(0, GAN_DIR)

from stable_baselines3 import PPO
from dataLoader import portfolio_load_dataset
from Environment.environment_IPM import PortfolioEnv as PortfolioEnvIPM
from IPM.ipm import IPMModule

# ── constantes ────────────────────────────────────────────────────────────────

ASSET_NAMES = ["SP500", "MSCI_EAFE", "MSCI_EM", "Gold", "Oil_WTI", "UST10Y", "Cash"]
SPLIT_DATE  = "2021-01-01"
FREQ_ANNUAL = 252   # días de trading para anualizar

# Episodios de mercado a identificar explícitamente
EVENTS = {
    "Rusia-Ucrania\n(feb-dic 2022)": ("2022-02-24", "2022-12-31"),
    "Crisis bancaria\n(mar-may 2023)": ("2023-03-08", "2023-05-31"),
}

# Umbrales VIX para régimen
VIX_MODERATE = 20
VIX_STRESS   = 30


# ── selección y carga de modelo ────────────────────────────────────────────────

def list_models(model_dir):
    return sorted(f[:-4] for f in os.listdir(model_dir) if f.endswith(".zip"))


def find_model(model_dir, model_name=None):
    available = list_models(model_dir)
    if not available:
        raise FileNotFoundError(f"No hay modelos .zip en {model_dir}")
    if model_name:
        if model_name not in available:
            raise FileNotFoundError(
                f"'{model_name}' no encontrado en {model_dir}\nDisponibles: {available}"
            )
        name = model_name
    else:
        # el más reciente
        name = sorted(
            available,
            key=lambda n: os.path.getmtime(os.path.join(model_dir, n + ".zip")),
            reverse=True,
        )[0]
    path = os.path.join(model_dir, name + ".zip")
    print(f"Modelo cargado: {path}")
    return PPO.load(path, device="cpu"), name


# ── datos ──────────────────────────────────────────────────────────────────────

def load_test_data(cache_dir):
    ds = portfolio_load_dataset(
        data_mode="Test",
        split_date=SPLIT_DATE,
        is_normalize=False,
        log_returns=True,
        use_intraday=True,
        use_windows=False,
    )
    features = ds.data[0, :, 0, :].T.astype(np.float32)   # (T, 18)

    prices_csv = os.path.join(cache_dir, "portfolio_prices_20040101_20251231.csv")
    if os.path.exists(prices_csv):
        all_dates = pd.read_csv(prices_csv, index_col=0, parse_dates=True).index
        test_dates = all_dates[all_dates >= pd.Timestamp(SPLIT_DATE)][1:]
        dates = pd.DatetimeIndex(test_dates[:len(features)])
    else:
        dates = pd.date_range(start=SPLIT_DATE, periods=len(features), freq="B")

    vix_path = os.path.join(cache_dir, "portfolio_vix_20040101_20251231.csv")
    vix_full = pd.read_csv(vix_path, index_col=0, parse_dates=True).iloc[:, 0]
    vix_test = vix_full.reindex(dates, method="ffill").fillna(20.0).values

    # alinear por el más corto
    n = min(len(features), len(vix_test), len(dates))
    features, vix_test, dates = features[:n], vix_test[:n], dates[:n]

    data = np.concatenate(
        [features, vix_test.reshape(-1, 1).astype(np.float32)], axis=1
    )
    return data, dates, vix_test


# ── evaluación ─────────────────────────────────────────────────────────────────

def _w(action):
    x = np.maximum(action, 0.0)
    s = x.sum()
    if s < 1e-8:
        w = np.zeros_like(x); w[-1] = 1.0; return w
    return x / s


def run_model(model, data, training_env=None):
    """Devuelve arrays (log_returns, weights) para todo el periodo.
    
    Args:
        model: El modelo PPO entrenado.
        data: Los datos del periodo de test.
        training_env: (Opcional) El entorno normalizado usado en el entrenamiento 
                      para copiar sus estadísticas obs_rms directamente.
    """
    ipm = IPMModule(m=18)
    # 1. Crear el entorno base de test
    raw_env = PortfolioEnvIPM(data, ipm_module=ipm, episode_weeks=None)
    
    # 2. Convertirlo a entorno vectorial (Obligatorio para usar VecNormalize)
    env = DummyVecEnv([lambda: raw_env])
    
    # 3. Aplicar el wrapper de normalización cargando las estadísticas
    # Si pasas el objeto del entrenamiento por argumento:
    if training_env is not None:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        env.obs_rms = training_env.obs_rms  # Copia matemática de la normalización
    else:
        # Si guardaste las estadísticas en un archivo .pkl durante el entrenamiento:
        env = VecNormalize.load("../RL/results_gan_augmented/vec_normalize_baseline.pkl", env)
    
    # 4. Congelar el entorno para el modo Test
    env.training = False     # Evita que las observaciones de test alteren la media/varianza aprendidas
    env.norm_reward = False  # Las recompensas no se normalizan en test para calcular métricas reales
    
    obs = env.reset()
    done = False
    lr, wh = [], []
    
    while not done:
        # deterministic=True apaga la entropía y el ruido SDE (ejecución pura)
        action, _ = model.predict(obs, deterministic=True)
        
        step_outputs = env.step(action)
        
        # Controlamos si el entorno es vectorial (4 salidas) o normal (5 salidas)
        if len(step_outputs) == 4:
            obs, reward, dones, infos = step_outputs
            info = infos[0] # Si es vectorial, extraemos el primer diccionario
            done = bool(dones)
        else:
            obs, reward, terminated, truncated, info = step_outputs
            # Si es un entorno normal, 'info' ya es el diccionario directo
            done = terminated or truncated
            
        # Extraemos el retorno de tu cartera de forma segura
        port_ret = float(info.get("portfolio_log_ret", reward))
        lr.append(port_ret)
        
    return np.array(lr), np.array(wh)


# ── métricas ───────────────────────────────────────────────────────────────────

METRIC_KEYS = ["n_decisiones", "ret_anualizado", "vol_anualizada",
               "sharpe", "sortino", "mdd", "var_95pct", "ret_total"]

def compute_metrics(lr, freq=FREQ_ANNUAL, var_conf=0.95):
    if len(lr) == 0:
        return {k: np.nan for k in METRIC_KEYS}
    mean    = lr.mean()
    std     = lr.std() + 1e-10
    # cada entrada cubre rebalance_freq días → anualizar en función de decisiones/año
    decisions_per_year = freq / 5
    sharpe  = np.sqrt(decisions_per_year) * mean / std
    down    = lr[lr < 0]
    sortino = np.sqrt(decisions_per_year) * mean / (np.sqrt((down**2).mean()) + 1e-10) if len(down) else np.inf
    eq      = np.concatenate([[1.0], np.exp(np.cumsum(lr))])
    peak    = np.maximum.accumulate(eq)
    mdd     = (eq / peak - 1).min()
    var     = np.percentile(lr, (1 - var_conf) * 100)
    total   = float(eq[-1] - 1)
    ann_ret = float((1 + total) ** (decisions_per_year / len(lr)) - 1)
    ann_vol = float(std * np.sqrt(decisions_per_year))
    return {
        "n_decisiones": len(lr),
        "ret_anualizado": ann_ret,
        "vol_anualizada": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "mdd": mdd,
        "var_95pct": var,
        "ret_total": total,
    }


# ── Diebold-Mariano ────────────────────────────────────────────────────────────

def diebold_mariano(lr1, lr2, h=5):
    """
    Contrasta H0: E[L(r1)] = E[L(r2)]  con pérdida L = -retorno.
    d_t > 0 implica que el modelo 2 supera al modelo 1 en t.
    Devuelve (estadístico DM, p-valor bilateral, media_diferencia).
    Usa corrección HAC de Harvey, Leybourne y Newbold (1997).
    """
    n = min(len(lr1), len(lr2))
    lr1, lr2 = lr1[:n], lr2[:n]
    d = lr2 - lr1                    # diferencial de pérdida (positivo = modelo2 mejor)
    d_bar = d.mean()
    T = len(d)

    # varianza HAC con h-1 retardos
    gamma = [np.mean((d - d_bar) * (np.roll(d, k) - d_bar)) for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / T

    if var_d <= 0:
        return np.nan, np.nan, d_bar

    dm_stat = d_bar / np.sqrt(var_d)

    # corrección de Harvey et al. (1997)
    correction = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    dm_stat_c  = dm_stat * correction
    p_value    = 2 * (1 - stats.t.cdf(abs(dm_stat_c), df=T - 1))

    return dm_stat_c, p_value, d_bar


# ── clasificación de regímenes y eventos ──────────────────────────────────────

def classify_regime(vix):
    regimes = np.where(vix < VIX_MODERATE, "low",
              np.where(vix < VIX_STRESS,   "moderate", "stress"))
    return regimes


def mask_event(dates, start_str, end_str):
    s = pd.Timestamp(start_str)
    e = pd.Timestamp(end_str)
    return (dates >= s) & (dates <= e)


# ── plots ──────────────────────────────────────────────────────────────────────

REGIME_COLORS = {"low": "#a8d5a2", "moderate": "#f9e08c", "stress": "#f4a56a"}
EVENT_COLORS  = {"Rusia-Ucrania\n(feb-dic 2022)": "#c0392b",
                 "Crisis bancaria\n(mar-may 2023)": "#8e44ad"}


def plot_equity_regimes(lr, dates, vix, event_lrs, model_name, out_path):
    eq     = np.concatenate([[1.0], np.exp(np.cumsum(lr))])
    peak   = np.maximum.accumulate(eq)
    dd     = eq / peak - 1
    regime = classify_regime(vix)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(f"Walk-forward 2021–2025 — {model_name}", fontsize=12, fontweight="bold")

    # ── equity con fondos de régimen ──
    ax = axes[0]
    for i in range(len(dates) - 1):
        ax.axvspan(dates[i], dates[i+1],
                   color=REGIME_COLORS.get(regime[i], "white"), alpha=0.35, linewidth=0)
    ax.plot(dates[:len(eq)-1], eq[1:], color="steelblue", linewidth=1.3, label=model_name)

    # eventos
    for evt_name, (s, e) in EVENTS.items():
        m = mask_event(dates, s, e)
        if m.any():
            ax.axvspan(dates[m][0], dates[m][-1],
                       color=EVENT_COLORS[evt_name], alpha=0.18, linewidth=0)
            mid = dates[m][len(dates[m])//2]
            ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] else 1,
                    evt_name, fontsize=7, ha="center", va="top", color=EVENT_COLORS[evt_name])

    # leyenda de regímenes
    patches = [mpatches.Patch(color=c, alpha=0.5, label=r.capitalize())
               for r, c in REGIME_COLORS.items()]
    ax.legend(handles=patches + [plt.Line2D([0],[0], color="steelblue", lw=1.3, label=model_name)],
              fontsize=8, loc="upper left")
    ax.set_ylabel("Equity (base 1.0)")
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")

    # ── drawdown ──
    ax = axes[1]
    ax.fill_between(dates[:len(dd)-1], dd[1:], 0, color="tomato", alpha=0.6)
    ax.axhline(dd.min(), color="darkred", linestyle="--", linewidth=0.8,
               label=f"MDD {dd.min():.2%}")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=8)

    # ── VIX ──
    ax = axes[2]
    ax.plot(dates, vix, color="gray", linewidth=0.8)
    ax.axhline(VIX_MODERATE, color="orange", linestyle="--", linewidth=0.7, label=f"VIX={VIX_MODERATE}")
    ax.axhline(VIX_STRESS,   color="red",    linestyle="--", linewidth=0.7, label=f"VIX={VIX_STRESS}")
    ax.set_ylabel("VIX")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Guardado → {out_path}")


def plot_metrics_table(rows_dict, title, out_path):
    df = pd.DataFrame(rows_dict).T
    numeric_cols = df.select_dtypes(include=np.number).columns

    fig, ax = plt.subplots(figsize=(max(10, len(df.columns) * 1.5), max(3, len(df) * 0.5 + 1.5)))
    ax.axis("off")

    def fmt(v):
        if isinstance(v, float):
            if abs(v) < 10:
                return f"{v:.3f}"
            return f"{v:.1f}"
        return str(v)

    cell_text = [[fmt(df.loc[idx, c]) for c in df.columns] for idx in df.index]
    tbl = ax.table(cellText=cell_text, rowLabels=list(df.index),
                   colLabels=list(df.columns), cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)

    # colorear por signo en Sharpe
    if "sharpe" in df.columns:
        for i, idx in enumerate(df.index):
            v = df.loc[idx, "sharpe"]
            if not np.isnan(v):
                color = "#d4edda" if v > 0 else "#f8d7da"
                tbl[(i+1, list(df.columns).index("sharpe"))].set_facecolor(color)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado → {out_path}")


def plot_dm_result(dm_stat, p_value, d_bar, model_name, baseline_name, out_path):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")

    significativo = p_value < 0.05 if not np.isnan(p_value) else False
    mejor = model_name if d_bar > 0 else baseline_name

    lines = [
        f"Test de Diebold-Mariano (Harvey, Leybourne y Newbold, 1997)",
        f"",
        f"H₀: no hay diferencia de rendimiento entre los modelos",
        f"",
        f"  Modelo principal :  {model_name}",
        f"  Baseline         :  {baseline_name}",
        f"",
        f"  Estadístico DM   :  {dm_stat:.4f}",
        f"  p-valor (bil.)   :  {p_value:.4f}",
        f"  Diferencia media :  {d_bar:.6f} retorno/día",
        f"",
        f"  {'✓ Diferencia significativa (p < 0.05)' if significativo else '✗ Diferencia NO significativa (p ≥ 0.05)'}",
        f"  {'→ ' + mejor + ' supera estadísticamente al otro modelo' if significativo else '→ No se puede rechazar H₀'}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.6",
                      facecolor="#d4edda" if significativo else "#fff3cd", alpha=0.9))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",     type=str, required=True,
                        help="Carpeta con el .zip del modelo principal")
    parser.add_argument("--model_name",    type=str, default=None,
                        help="Nombre del modelo sin extensión (por defecto el más reciente)")
    parser.add_argument("--baseline_dir",  type=str, default=None,
                        help="Carpeta con el modelo baseline (para test DM)")
    parser.add_argument("--baseline_name", type=str, default=None,
                        help="Nombre del baseline sin extensión")
    parser.add_argument("--cache_dir",     type=str,
                        default=os.path.join(RL_DIR, "data_cache"))
    parser.add_argument("--list",          action="store_true",
                        help="Listar modelos disponibles y salir")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("Modelos disponibles:")
        for m in list_models(args.model_dir):
            print(f"  {m}")
        return

    out_dir = args.model_dir
    os.makedirs(out_dir, exist_ok=True)

    # Cargar modelo principal
    model, model_name = find_model(args.model_dir, args.model_name)

    # Datos de test
    print("Cargando datos 2021+...")
    data, dates, vix = load_test_data(args.cache_dir)
    T = len(data)
    print(f"  Período: {dates[0].date()} → {dates[-1].date()}  ({T} días de trading)")

    # Evaluación
    print("Evaluando modelo principal...")
    lr, weights = run_model(model, data)

    # Cada decisión cubre 5 días de trading; el entorno empieza en el paso 5
    # → fecha de la k-ésima decisión = dates[5 + k*5]
    WINDOW = 5
    decision_idx = np.arange(WINDOW, WINDOW + len(lr) * WINDOW, WINDOW)
    decision_idx = decision_idx[decision_idx < T]
    n = len(decision_idx)
    lr, weights = lr[:n], weights[:n]
    dates_lr = dates[decision_idx]
    vix_lr   = vix[decision_idx]

    # ── Métricas globales ──────────────────────────────────────────────
    rows = {}
    rows["GLOBAL 2021–2025"] = compute_metrics(lr)

    # ── Por régimen VIX ───────────────────────────────────────────────
    regime = classify_regime(vix_lr)
    for reg in ["low", "moderate", "stress"]:
        mask = regime == reg
        rows[f"Régimen {reg.capitalize()} (VIX {'<20' if reg=='low' else '20-30' if reg=='moderate' else '>30'})"] \
            = compute_metrics(lr[mask])

    # ── Por año ───────────────────────────────────────────────────────
    years = dates_lr.year
    for yr in sorted(set(years)):
        mask = years == yr
        rows[str(yr)] = compute_metrics(lr[mask])

    # ── Episodios explícitos ───────────────────────────────────────────
    for evt_name, (s, e) in EVENTS.items():
        mask = mask_event(dates_lr, s, e)
        rows[evt_name.replace("\n", " ")] = compute_metrics(lr[mask])

    df_metrics = pd.DataFrame(rows).T
    csv_path = os.path.join(out_dir, f"wf_{model_name}_metrics.csv")
    df_metrics.to_csv(csv_path, float_format="%.4f")
    print(f"\nCSV métricas → {csv_path}")
    print(df_metrics.to_string())

    # ── Plots ──────────────────────────────────────────────────────────
    plot_equity_regimes(
        lr, dates_lr, vix_lr, {},
        model_name,
        os.path.join(out_dir, f"wf_{model_name}_equity.png"),
    )
    plot_metrics_table(
        rows,
        f"Métricas walk-forward 2021–2025 — {model_name}",
        os.path.join(out_dir, f"wf_{model_name}_tabla.png"),
    )

    # ── Test de Diebold-Mariano ────────────────────────────────────────
    if args.baseline_dir:
        print("\nCargando modelo baseline para test DM...")
        baseline, baseline_name = find_model(args.baseline_dir, args.baseline_name)
        print("Evaluando baseline...")
        lr_base, _ = run_model(baseline, data)
        lr_base = lr_base[:n]   # alinear al mismo número de decisiones

        dm_stat, p_value, d_bar = diebold_mariano(lr_base, lr)
        print(f"\nDiebold-Mariano → DM={dm_stat:.4f}  p={p_value:.4f}  Δmedia={d_bar:.6f}")

        plot_dm_result(
            dm_stat, p_value, d_bar, model_name, baseline_name,
            os.path.join(out_dir, f"wf_dm_{model_name}_vs_{baseline_name}.png"),
        )

        # tabla comparativa
        rows_base = {"GLOBAL": compute_metrics(lr_base)}
        for reg in ["low", "moderate", "stress"]:
            mask = regime == reg
            rows_base[f"Régimen {reg}"] = compute_metrics(lr_base[mask])
        for yr in sorted(set(years)):
            rows_base[str(yr)] = compute_metrics(lr_base[years == yr])
        for evt_name, (s, e) in EVENTS.items():
            mask = mask_event(dates_lr, s, e)
            rows_base[evt_name.replace("\n", " ")] = compute_metrics(lr_base[mask])

        df_comp = pd.DataFrame({
            **{f"{k} [{model_name}]": v for k, v in rows.items()},
            **{f"{k} [{baseline_name}]": v for k, v in rows_base.items()},
        }).T
        comp_csv = os.path.join(out_dir, f"wf_comparison_{model_name}_vs_{baseline_name}.csv")
        df_comp.to_csv(comp_csv, float_format="%.4f")
        print(f"CSV comparativo → {comp_csv}")


if __name__ == "__main__":
    main()