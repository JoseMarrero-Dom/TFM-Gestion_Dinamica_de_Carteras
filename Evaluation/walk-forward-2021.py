"""
Walk-forward backtesting 2021–2025 del agente PPO.

Métricas: Sharpe (1994), Sortino (1994), Calmar, MDD, VaR/CVaR al 95%,
hit rate y turnover — desagregadas por subperiodo de régimen VIX y por
episodios explícitos (guerra Rusia-Ucrania 2022, crisis bancaria 2023).

Uso:
    # listar modelos disponibles
    python walk-forward-2021.py --model_dir ../RL/results_gan_augmented --list

    # evaluar un modelo concreto
    python walk-forward-2021.py --model_dir ../RL/results_gan_augmented \\
                                --model_name ppo_augmented
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    raise FileNotFoundError(
        f"No se encontró un archivo VecNormalize en {model_dir}. "
        f"Probados: {candidates}"
    )


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


def run_model(model, data, vecnormalize_path=None, training_env=None):
    """Devuelve arrays (log_returns, weights, rebalance_freq) para todo el periodo.

    Args:
        model: El modelo PPO entrenado.
        data: Los datos del periodo de test.
        training_env: (Opcional) El entorno normalizado usado en el entrenamiento
                      para copiar sus estadísticas obs_rms directamente.
    """
    ipm = IPMModule(m=18)
    raw_env = PortfolioEnvIPM(data, ipm_module=ipm, episode_weeks=None)
    rebalance_freq = raw_env.rebalance_freq

    env = DummyVecEnv([lambda: raw_env])
    if training_env is not None:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        env.obs_rms = training_env.obs_rms
    else:
        if vecnormalize_path is None:
            raise ValueError("Debes proporcionar vecnormalize_path cuando training_env is None.")
        env = VecNormalize.load(vecnormalize_path, env)

    env.training    = False
    env.norm_reward = False

    obs = env.reset()
    done = False
    lr, wh = [], []

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
        lr.append(port_ret)
        # capturamos los pesos post-drift del entorno (no del wrapper) para turnover
        try:
            wh.append(env.get_attr("weights")[0].copy())
        except Exception:
            pass

    return np.array(lr), np.array(wh), rebalance_freq


# ── métricas ───────────────────────────────────────────────────────────────────

METRIC_KEYS = ["n_decisiones", "ret_anualizado", "vol_anualizada",
               "sharpe", "sortino", "mdd", "var_95pct", "ret_total"]


def compute_metrics(lr, freq=FREQ_ANNUAL, rebalance_freq=5, var_conf=0.95):
    """Métricas de cartera sobre log-returns por decisión.

    Convenciones:
      - Sharpe (Sharpe, 1994):  √(decisiones/año) · mean / std  con rf=0.
        Es un ratio; se anualiza por convención académica aunque el periodo
        sea más corto que un año.
      - Sortino (Sortino & Price, 1994): downside deviation calculada con
        sqrt(mean(min(r,0)^2)) sobre todo el periodo, no solo los negativos
        (la versión "solo negativos" sobreestima sistemáticamente el ratio).
      - ret_anualizado y vol_anualizada: solo tienen sentido cuando el
        periodo evaluado cubre al menos un año hábil. Para subperiodos más
        cortos (eventos, regímenes con pocos días) se devuelven NaN para
        no extrapolar una rentabilidad anual a partir de pocos datos.
      - VaR histórico al nivel var_conf.
    """
    if len(lr) == 0:
        return {k: np.nan for k in METRIC_KEYS}

    decisions_per_year = freq / rebalance_freq
    # un año natural de mercado da exactamente int(decisions_per_year)=50
    # decisiones (252/5=50.4 → 50 enteras). Pedir 252 días estrictos dejaría
    # fuera los años completos por el redondeo.
    is_annualizable = len(lr) >= int(decisions_per_year)

    mean = lr.mean()
    std  = lr.std() + 1e-10

    sharpe = np.sqrt(decisions_per_year) * mean / std

    downside = np.minimum(lr, 0.0)
    dd_dev   = np.sqrt(np.mean(downside ** 2))
    sortino  = np.sqrt(decisions_per_year) * mean / dd_dev if dd_dev > 1e-10 else np.nan

    eq   = np.concatenate([[1.0], np.exp(np.cumsum(lr))])
    peak = np.maximum.accumulate(eq)
    mdd  = float((eq / peak - 1).min())

    q   = (1 - var_conf) * 100
    var = float(np.percentile(lr, q))

    total = float(eq[-1] - 1)

    if is_annualizable and total > -1:
        ann_ret = float((1 + total) ** (decisions_per_year / len(lr)) - 1)
        ann_vol = float(std * np.sqrt(decisions_per_year))
    else:
        ann_ret = np.nan
        ann_vol = np.nan

    return {
        "n_decisiones":  len(lr),
        "ret_anualizado": ann_ret,
        "vol_anualizada": ann_vol,
        "sharpe":  sharpe,
        "sortino": sortino,
        "mdd":     mdd,
        "var_95pct":  var,
        "ret_total":  total,
    }


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


def plot_portfolio_boxplot(weights, asset_labels, title, out_path):
    """Box-plot de la composición de cartera: una caja por activo (+ cash)
    mostrando la distribución de los pesos asignados a lo largo del periodo.
    """
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(asset_labels) + 1), 5))
    cols = [weights[:, i] for i in range(weights.shape[1])]
    bp = ax.boxplot(cols, tick_labels=asset_labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black",
                                   markeredgecolor="black", markersize=4),
                    medianprops=dict(color="black", linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#7f7f7f"]
    for patch, c in zip(bp["boxes"], palette[:len(cols)]):
        patch.set_facecolor(c); patch.set_alpha(0.6)

    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Peso asignado")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax.tick_params(axis="x", labelsize=8, pad=6)
    for i, col in enumerate(cols, start=1):
        med = np.median(col)
        mean = np.mean(col)
        ax.text(i, -0.09, f"Mediana={med:.2f}\nMedia={mean:.2f}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, color="dimgray")

    plt.subplots_adjust(bottom=0.30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado → {out_path}")


def plot_portfolio_boxplots_by_year(weights, dates, asset_labels, model_name, out_dir):
    """Genera una gráfica por año (más la global) con la distribución de
    pesos por activo. Cada gráfica es un PNG independiente.
    """
    if len(weights) == 0:
        print("Sin pesos capturados, no se pueden hacer box-plots de cartera.")
        return

    n = min(len(weights), len(dates))
    weights = weights[:n]; dates = dates[:n]

    plot_portfolio_boxplot(
        weights, asset_labels,
        f"Distribución de cartera — Global 2021–2025 — {model_name}",
        os.path.join(out_dir, f"wf_{model_name}_cartera_global.png"),
    )
    for yr in sorted(set(dates.year)):
        mask = dates.year == yr
        if mask.sum() == 0:
            continue
        plot_portfolio_boxplot(
            weights[mask], asset_labels,
            f"Distribución de cartera — {yr} — {model_name}",
            os.path.join(out_dir, f"wf_{model_name}_cartera_{yr}.png"),
        )


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


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",     type=str, required=True,
                        help="Carpeta con el .zip del modelo principal")
    parser.add_argument("--model_name",    type=str, default=None,
                        help="Nombre del modelo sin extensión (por defecto el más reciente)")
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
    vecnormalize_path = find_vec_normalize(args.model_dir, model_name)

    # Datos de test
    print("Cargando datos 2021+...")
    data, dates, vix = load_test_data(args.cache_dir)
    T = len(data)
    print(f"  Período: {dates[0].date()} → {dates[-1].date()}  ({T} días de trading)")

    # Evaluación
    print("Evaluando modelo principal...")
    lr, weights, rebal = run_model(model, data, vecnormalize_path=vecnormalize_path)

    # Cada decisión cubre rebal días → fecha de la k-ésima decisión = dates[rebal + k*rebal]
    decision_idx = np.arange(rebal, rebal + len(lr) * rebal, rebal)
    decision_idx = decision_idx[decision_idx < T]
    n = len(decision_idx)
    lr, weights = lr[:n], weights[:n] if len(weights) else weights
    dates_lr = dates[decision_idx]
    vix_lr   = vix[decision_idx]

    def _metrics(arr):
        return compute_metrics(arr, freq=FREQ_ANNUAL, rebalance_freq=rebal)

    # ── Métricas globales ──────────────────────────────────────────────
    rows = {}
    rows["GLOBAL 2021–2025"] = _metrics(lr)

    # ── Por régimen VIX ───────────────────────────────────────────────
    regime = classify_regime(vix_lr)
    for reg in ["low", "moderate", "stress"]:
        mask = regime == reg
        rows[f"Régimen {reg.capitalize()} (VIX {'<20' if reg=='low' else '20-30' if reg=='moderate' else '>30'})"] \
            = _metrics(lr[mask])

    # ── Por año ───────────────────────────────────────────────────────
    years = dates_lr.year
    for yr in sorted(set(years)):
        mask = years == yr
        rows[str(yr)] = _metrics(lr[mask])

    # ── Episodios explícitos ───────────────────────────────────────────
    for evt_name, (s, e) in EVENTS.items():
        mask = mask_event(dates_lr, s, e)
        rows[evt_name.replace("\n", " ")] = _metrics(lr[mask])

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
    plot_portfolio_boxplots_by_year(
        weights, dates_lr, ASSET_NAMES, model_name, out_dir,
    )


if __name__ == "__main__":
    main()