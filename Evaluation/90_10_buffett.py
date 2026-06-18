'''
Cartera 90/10 Warren Buffett de 2021, con los mismos tickers que la cartera dinámica, pero sin reequilibrar. Con esta cartera se pueden comparar los resultados de la GAN con una estrategia buy&hold.
Con una distribución de persos de 90% en el sp500 y 10% en bonos gubernamentales.
'''

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib.util


GAN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../TTS-GAN/tts-gan-tfm"))
DATA_LOADER_PATH = os.path.join(GAN_DIR, "dataLoader.py")

spec = importlib.util.spec_from_file_location("dataLoader", DATA_LOADER_PATH)
if spec is None or spec.loader is None:
	raise ImportError(f"No se pudo cargar dataLoader desde {DATA_LOADER_PATH}")
data_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_loader)

DEFAULT_TICKERS = data_loader.DEFAULT_TICKERS
portfolio_load_dataset = data_loader.portfolio_load_dataset


ASSET_NAMES = list(DEFAULT_TICKERS.keys())
SPLIT_DATE = "2021-01-01"
FREQ_ANNUAL = 252
VIX_MODERATE = 20
VIX_STRESS = 30
EVENTS = {
	"Rusia-Ucrania\n(feb-dic 2022)": ("2022-02-24", "2022-12-31"),
	"Crisis bancaria\n(mar-may 2023)": ("2023-03-08", "2023-05-31"),
}
REGIME_COLORS = {"low": "#a8d5a2", "moderate": "#f9e08c", "stress": "#f4a56a"}
EVENT_COLORS = {
	"Rusia-Ucrania\n(feb-dic 2022)": "#c0392b",
	"Crisis bancaria\n(mar-may 2023)": "#8e44ad",
}


def parse_args():
	parser = argparse.ArgumentParser(description="Cartera 90/10 Warren Buffett 2021 con pesos uniformes.")
	parser.add_argument(
		"--cache_dir",
		type=str,
		default=os.path.join(GAN_DIR, "data_cache"),
		help="Carpeta con los CSV cacheados de precios y VIX.",
	)
	parser.add_argument(
		"--out_dir",
		type=str,
		default=os.path.join(os.path.dirname(__file__), "results_90_10_2021"),
		help="Carpeta donde guardar métricas y gráficas.",
	)
	return parser.parse_args()


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
	features = ds.data[0, :, 0, :].T.astype(np.float32)  # (T, 18)

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
	return data, dates, vix_test


def classify_regime(vix):
	return np.where(vix < VIX_MODERATE, "low", np.where(vix < VIX_STRESS, "moderate", "stress"))


def mask_event(dates, start_str, end_str):
	s = pd.Timestamp(start_str)
	e = pd.Timestamp(end_str)
	return (dates >= s) & (dates <= e)


def safe_name(value):
	return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def simulate_buy_and_hold(asset_log_returns, initial_weights):
	"""Simula una cartera buy&hold sin reequilibrado."""
	asset_values = initial_weights[None, :] * np.exp(np.cumsum(asset_log_returns, axis=0))
	portfolio_wealth = asset_values.sum(axis=1)
	log_returns = np.diff(np.log(np.concatenate([[1.0], portfolio_wealth])))
	weights = asset_values / portfolio_wealth[:, None]
	return log_returns.astype(np.float32), weights.astype(np.float32)


def compute_metrics(lr, freq=FREQ_ANNUAL, var_conf=0.95):
	if len(lr) == 0:
		return {
			"n_dias": 0,
			"ret_anualizado": np.nan,
			"vol_anualizada": np.nan,
			"sharpe": np.nan,
			"sortino": np.nan,
			"mdd": np.nan,
			"var_95pct": np.nan,
			"ret_total": np.nan,
		}

	mean = lr.mean()
	std = lr.std() + 1e-10
	sharpe = np.sqrt(freq) * mean / std

	downside = np.minimum(lr, 0.0)
	dd_dev = np.sqrt(np.mean(downside**2))
	sortino = np.sqrt(freq) * mean / dd_dev if dd_dev > 1e-10 else np.nan

	eq = np.concatenate([[1.0], np.exp(np.cumsum(lr))])
	peak = np.maximum.accumulate(eq)
	mdd = float((eq / peak - 1).min())

	var = float(np.percentile(lr, (1 - var_conf) * 100))
	total = float(eq[-1] - 1)

	if len(lr) >= freq and total > -1:
		ann_ret = float((1 + total) ** (freq / len(lr)) - 1)
		ann_vol = float(std * np.sqrt(freq))
	else:
		ann_ret = np.nan
		ann_vol = np.nan

	return {
		"n_dias": len(lr),
		"ret_anualizado": ann_ret,
		"vol_anualizada": ann_vol,
		"sharpe": sharpe,
		"sortino": sortino,
		"mdd": mdd,
		"var_95pct": var,
		"ret_total": total,
	}


def plot_equity_regimes(lr, dates, vix, model_name, out_path):
	eq = np.concatenate([[1.0], np.exp(np.cumsum(lr))])
	peak = np.maximum.accumulate(eq)
	dd = eq / peak - 1
	regime = classify_regime(vix)

	fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
	fig.suptitle(f"Cartera 90/10 Warren Buffett 2021–2025 — {model_name}", fontsize=12, fontweight="bold")

	ax = axes[0]
	for i in range(len(dates) - 1):
		ax.axvspan(dates[i], dates[i + 1], color=REGIME_COLORS.get(regime[i], "white"), alpha=0.35, linewidth=0)
	ax.plot(dates[: len(eq) - 1], eq[1:], color="steelblue", linewidth=1.3, label=model_name)

	for evt_name, (s, e) in EVENTS.items():
		m = mask_event(dates, s, e)
		if m.any():
			ax.axvspan(dates[m][0], dates[m][-1], color=EVENT_COLORS[evt_name], alpha=0.18, linewidth=0)
			mid = dates[m][len(dates[m]) // 2]
			ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] else 1, evt_name, fontsize=7, ha="center", va="top", color=EVENT_COLORS[evt_name])

	patches = [mpatches.Patch(color=c, alpha=0.5, label=r.capitalize()) for r, c in REGIME_COLORS.items()]
	ax.legend(handles=patches + [plt.Line2D([0], [0], color="steelblue", lw=1.3, label=model_name)], fontsize=8, loc="upper left")
	ax.set_ylabel("Equity (base 1.0)")
	ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")

	ax = axes[1]
	ax.fill_between(dates[: len(dd) - 1], dd[1:], 0, color="tomato", alpha=0.6)
	ax.axhline(dd.min(), color="darkred", linestyle="--", linewidth=0.8, label=f"MDD {dd.min():.2%}")
	ax.set_ylabel("Drawdown")
	ax.legend(fontsize=8)

	ax = axes[2]
	ax.plot(dates, vix, color="gray", linewidth=0.8)
	ax.axhline(VIX_MODERATE, color="orange", linestyle="--", linewidth=0.7, label=f"VIX={VIX_MODERATE}")
	ax.axhline(VIX_STRESS, color="red", linestyle="--", linewidth=0.7, label=f"VIX={VIX_STRESS}")
	ax.set_ylabel("VIX")
	ax.legend(fontsize=7)

	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"Guardado → {out_path}")


def plot_portfolio_boxplot(weights, asset_labels, title, out_path):
	fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(asset_labels) + 1), 5))
	cols = [weights[:, i] for i in range(weights.shape[1])]
	bp = ax.boxplot(
		cols,
		labels=asset_labels,
		patch_artist=True,
		showmeans=True,
		meanprops=dict(marker="D", markerfacecolor="black", markeredgecolor="black", markersize=4),
		medianprops=dict(color="black", linewidth=1.2),
		flierprops=dict(marker="o", markersize=3, alpha=0.5),
	)

	palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
	for patch, c in zip(bp["boxes"], palette[: len(cols)]):
		patch.set_facecolor(c)
		patch.set_alpha(0.6)

	ax.set_ylim(-0.02, 1.02)
	ax.set_ylabel("Peso asignado")
	ax.set_title(title, fontsize=11, fontweight="bold")
	ax.grid(axis="y", alpha=0.3)

	for i, col in enumerate(cols, start=1):
		med = np.median(col)
		mean = np.mean(col)
		ax.text(i, 1.0, f"med={med:.2f}\nμ={mean:.2f}", ha="center", va="top", fontsize=7, color="dimgray")

	plt.tight_layout()
	plt.savefig(out_path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"Guardado → {out_path}")


def plot_portfolio_boxplots_by_year(weights, dates, asset_labels, model_name, out_dir):
	if len(weights) == 0:
		print("Sin pesos capturados, no se pueden hacer box-plots de cartera.")
		return

	n = min(len(weights), len(dates))
	weights = weights[:n]
	dates = dates[:n]
	file_stub = safe_name(model_name)

	plot_portfolio_boxplot(
		weights,
		asset_labels,
		f"Distribución de cartera — Global 2021–2025 — {model_name}",
		os.path.join(out_dir, f"static_{file_stub}_cartera_global.png"),
	)
	for yr in sorted(set(dates.year)):
		mask = dates.year == yr
		if mask.sum() == 0:
			continue
		plot_portfolio_boxplot(
			weights[mask],
			asset_labels,
			f"Distribución de cartera — {yr} — {model_name}",
			os.path.join(out_dir, f"static_{file_stub}_cartera_{yr}.png"),
		)


def plot_metrics_table(rows_dict, title, out_path):
	df = pd.DataFrame(rows_dict).T

	fig, ax = plt.subplots(figsize=(max(10, len(df.columns) * 1.5), max(3, len(df) * 0.5 + 1.5)))
	ax.axis("off")

	def fmt(v):
		if isinstance(v, float):
			if np.isnan(v):
				return "nan"
			if abs(v) < 10:
				return f"{v:.3f}"
			return f"{v:.1f}"
		return str(v)

	cell_text = [[fmt(df.loc[idx, c]) for c in df.columns] for idx in df.index]
	tbl = ax.table(cellText=cell_text, rowLabels=list(df.index), colLabels=list(df.columns), cellLoc="center", loc="center")
	tbl.auto_set_font_size(False)
	tbl.set_fontsize(8)
	tbl.scale(1, 1.4)

	if "sharpe" in df.columns:
		for i, idx in enumerate(df.index):
			v = df.loc[idx, "sharpe"]
			if not np.isnan(v):
				color = "#d4edda" if v > 0 else "#f8d7da"
				tbl[(i + 1, list(df.columns).index("sharpe"))].set_facecolor(color)

	ax.set_title(title, fontsize=10, fontweight="bold", pad=12)
	plt.tight_layout()
	plt.savefig(out_path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"Guardado → {out_path}")


def main():
	args = parse_args()
	os.makedirs(args.out_dir, exist_ok=True)

	print("Cargando datos 2021+...")
	data, dates, vix = load_test_data(args.cache_dir)
	features = data[:, :18]
	asset_lr = features[:, ::3]  # close-to-close por activo
	print(f"  Período: {dates[0].date()} → {dates[-1].date()}  ({len(dates)} días de trading)")

	initial_weights = np.array([0.9] + [0.0] * (len(ASSET_NAMES) - 2) + [0.1], dtype=np.float32)
	lr, weights = simulate_buy_and_hold(asset_lr, initial_weights)
	n = min(len(lr), len(dates), len(vix))
	lr = lr[:n]
	weights = weights[:n]
	dates_lr = dates[:n]
	vix_lr = vix[:n]

	rows = {}
	rows["GLOBAL 2021–2025"] = compute_metrics(lr)

	regime = classify_regime(vix_lr)
	for reg in ["low", "moderate", "stress"]:
		mask = regime == reg
		rows[f"Régimen {reg.capitalize()} (VIX {'<20' if reg == 'low' else '20-30' if reg == 'moderate' else '>30'})"] = compute_metrics(lr[mask])

	years = dates_lr.year
	for yr in sorted(set(years)):
		mask = years == yr
		rows[str(yr)] = compute_metrics(lr[mask])

	for evt_name, (s, e) in EVENTS.items():
		mask = mask_event(dates_lr, s, e)
		rows[evt_name.replace("\n", " ")] = compute_metrics(lr[mask])

	df_metrics = pd.DataFrame(rows).T
	csv_path = os.path.join(args.out_dir, "static_2021_metrics.csv")
	df_metrics.to_csv(csv_path, float_format="%.4f")
	print(f"\nCSV métricas → {csv_path}")
	print(df_metrics.to_string())

	plot_equity_regimes(lr, dates_lr, vix_lr, "Cartera 90/10 Warren Buffett 2021", os.path.join(args.out_dir, "static_2021_equity.png"))
	plot_metrics_table(rows, "Métricas cartera 90/10 Warren Buffett 2021–2025", os.path.join(args.out_dir, "static_2021_tabla.png"))
	plot_portfolio_boxplots_by_year(weights, dates_lr, ASSET_NAMES, "Cartera 90/10 Warren Buffett 2021", args.out_dir)


if __name__ == "__main__":
	main()