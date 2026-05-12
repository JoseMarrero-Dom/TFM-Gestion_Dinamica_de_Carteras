# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from torch.utils.data import Dataset


DEFAULT_TICKERS: Dict[str, str] = {
	"SP500": "^GSPC",
	"MSCI_EAFE": "EFA",
	"MSCI_EM": "EEM",
	"Gold": "GLD",
	"Oil_WTI": "USO",
	"UST10Y": "IEF",
}

DEFAULT_VIX_TICKER = "^VIX"

REGIME_LABELS = ("low", "moderate", "stress")
REGIME_TO_ID = {"low": 0, "moderate": 1, "stress": 2}


class portfolio_load_dataset(Dataset):
	def __init__(
		self,
		data_mode: str = "Train",
		start: str = "2004-01-01",
		end: str = "2025-12-31",
		tickers: Optional[Dict[str, str]] = None,
		assets: Optional[List[str]] = None,
		window_length: int = 150,
		stride: int = 1,
		train_ratio: float = 0.8,
		split_date: Optional[str] = None,
		log_returns: bool = True,
		is_normalize: bool = True,
		normalize_mode: str = "zscore",
		label_mode: str = "dummy",
		filter_regime: Optional[List[str]] = None,
		include_vix: bool = False,
		use_intraday: bool = False,
		cache_dir: str = "./data_cache",
		cache_prefix: str = "portfolio",
		use_windows: bool = True,
		verbose: bool = False,
	) -> None:
		if data_mode not in ("Train", "Test"):
			raise ValueError("data_mode must be 'Train' or 'Test'.")
		if window_length <= 1:
			raise ValueError("window_length must be > 1.")
		if stride <= 0:
			raise ValueError("stride must be > 0.")
		if label_mode not in ("dummy", "regime"):
			raise ValueError("label_mode must be 'dummy' or 'regime'.")
		if filter_regime is not None and not all(r in REGIME_LABELS for r in filter_regime):
			raise ValueError("filter_regime must be a list containing only: low, moderate, stress.")

		self.data_mode = data_mode
		self.start = start
		self.end = end
		self.tickers = tickers or DEFAULT_TICKERS
		self.assets = assets or list(self.tickers.keys())
		self.window_length = window_length
		self.stride = stride
		self.train_ratio = train_ratio
		self.split_date = split_date
		self.log_returns = log_returns
		self.is_normalize = is_normalize
		self.normalize_mode = normalize_mode
		self.label_mode = label_mode
		self.filter_regime = filter_regime
		self.use_intraday = use_intraday
		self.include_vix = include_vix or (label_mode == "regime") or (filter_regime is not None)
		self.cache_dir = cache_dir
		self.cache_prefix = cache_prefix
		self.use_windows = use_windows
		self.verbose = verbose

		prices, open_prices, high_prices, low_prices, vix = self._load_prices_and_vix()
		returns = self._compute_returns(prices, open_prices, high_prices, low_prices)

		train_df, test_df = self._split_data(returns)
		df = train_df if self.data_mode == "Train" else test_df

		regime_series = None
		if self.include_vix:
			vix = vix.reindex(returns.index).ffill()
			regime_series = self._classify_regime(vix)
			regime_series = regime_series.reindex(df.index)
	
		if self.use_windows:
			windows, labels = self._build_windows(df, regime_series)
		else:
			values = df.values.astype(np.float32)        # (T, m)
			windows = values[None, :, :]                 # (1, T, m)
			labels = np.zeros((1,), dtype=np.int64)      # dummy

		if self.is_normalize:
			windows = self._normalize_windows(windows)

		self.data = self._format_windows(windows)
		self.labels = labels

		if self.verbose:
			print(f"windows shape: {self.data.shape}")
			print(f"labels shape: {self.labels.shape}")

	def _load_prices_and_vix(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
		os.makedirs(self.cache_dir, exist_ok=True)
		safe_range = f"{self.start}_{self.end}".replace("-", "")
		prices_path = os.path.join(self.cache_dir, f"{self.cache_prefix}_prices_{safe_range}.csv")
		open_path   = os.path.join(self.cache_dir, f"{self.cache_prefix}_open_{safe_range}.csv")
		high_path   = os.path.join(self.cache_dir, f"{self.cache_prefix}_high_{safe_range}.csv")
		low_path    = os.path.join(self.cache_dir, f"{self.cache_prefix}_low_{safe_range}.csv")
		vix_path    = os.path.join(self.cache_dir, f"{self.cache_prefix}_vix_{safe_range}.csv")

		def _download_col(col: str, path: str) -> pd.DataFrame:
			if os.path.isfile(path):
				return pd.read_csv(path, index_col=0, parse_dates=True)
			raw = {}
			for name, ticker in self.tickers.items():
				data = yf.download(ticker, start=self.start, end=self.end, auto_adjust=True, progress=False)
				if data.empty or col not in data.columns:
					raise RuntimeError(f"No '{col}' data for ticker {ticker}.")
				raw[name] = data[col].squeeze()
			df = pd.DataFrame(raw).dropna()
			df.to_csv(path)
			return df

		prices      = _download_col("Close", prices_path)
		open_prices = _download_col("Open",  open_path)
		high_prices = _download_col("High",  high_path)
		low_prices  = _download_col("Low",   low_path)

		if os.path.isfile(vix_path):
			vix = pd.read_csv(vix_path, index_col=0, parse_dates=True).iloc[:, 0]
		else:
			vix_raw = yf.download(DEFAULT_VIX_TICKER, start=self.start, end=self.end, auto_adjust=True, progress=False)
			if vix_raw.empty or "Close" not in vix_raw.columns:
				vix = pd.Series(index=prices.index, data=np.nan)
			else:
				vix = vix_raw["Close"].squeeze()
			vix.to_csv(vix_path)

		return prices[self.assets], open_prices[self.assets], high_prices[self.assets], low_prices[self.assets], vix

	def _compute_returns(
		self,
		prices: pd.DataFrame,
		open_prices: pd.DataFrame,
		high_prices: pd.DataFrame,
		low_prices: pd.DataFrame,
	) -> pd.DataFrame:
		if not self.use_intraday:
			if self.log_returns:
				returns = np.log(prices / prices.shift(1))
			else:
				returns = prices.pct_change()
			returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
			return returns

		# Canal 1: log return close-to-close — dirección y magnitud del día
		log_ret = np.log(prices / prices.shift(1))
		# Canal 2: rango open-close — si el día cerró alcista o bajista
		oc_range = np.log(prices / open_prices.reindex(prices.index))
		# Canal 3: rango high-low — volatilidad intradía (siempre positivo)
		hl_range = np.log(high_prices.reindex(prices.index) / low_prices.reindex(prices.index))

		log_ret.columns  = [f"{c}_logret"   for c in log_ret.columns]
		oc_range.columns = [f"{c}_oc_range"  for c in oc_range.columns]
		hl_range.columns = [f"{c}_hl_range"  for c in hl_range.columns]

		# Intercalar por activo: SP500_logret, SP500_oc_range, SP500_hl_range, ...
		combined = pd.concat([log_ret, oc_range, hl_range], axis=1)
		ordered_cols = [col for a in self.assets for col in (f"{a}_logret", f"{a}_oc_range", f"{a}_hl_range")]
		combined = combined[ordered_cols]

		combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
		return combined

	def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
		if self.split_date:
			split_ts = pd.Timestamp(self.split_date)
			train_df = df.loc[df.index < split_ts]
			test_df = df.loc[df.index >= split_ts]
		else:
			split_idx = int(len(df) * self.train_ratio)
			train_df = df.iloc[:split_idx]
			test_df = df.iloc[split_idx:]
		if train_df.empty or test_df.empty:
			raise RuntimeError("Train/test split resulted in empty data.")
		return train_df, test_df

	def _classify_regime(self, vix: pd.Series) -> pd.Series:
		def _label(val: float) -> str:
			if val < 20:
				return "low"
			if val < 30:
				return "moderate"
			return "stress"

		return vix.map(_label)

	def _build_windows(
		self,
		df: pd.DataFrame,
		regime_series: Optional[pd.Series],
	) -> Tuple[np.ndarray, np.ndarray]:
		values = df.values.astype(np.float32)
		windows: List[np.ndarray] = []
		labels: List[int] = []

		for start in range(0, len(values) - self.window_length + 1, self.stride):
			end = start + self.window_length
			window = values[start:end]
			if regime_series is not None:
				end_date = df.index[end - 1]
				regime = regime_series.loc[end_date]
				if self.filter_regime is not None and regime not in self.filter_regime:
					continue
				if self.label_mode == "regime":
					labels.append(REGIME_TO_ID[regime])
			windows.append(window)

		if not windows:
			raise RuntimeError("No windows created. Check window_length/stride and filters.")

		if self.label_mode == "dummy":
			labels = [0 for _ in range(len(windows))]
		labels_array = np.asarray(labels, dtype=np.int64)
		windows_array = np.stack(windows, axis=0)
		return windows_array, labels_array

	def _normalize_windows(self, windows: np.ndarray) -> np.ndarray:
		if self.normalize_mode == "zscore":
			mean = windows.mean(axis=1, keepdims=True)
			std = windows.std(axis=1, keepdims=True) + 1e-8
			return (windows - mean) / std
		if self.normalize_mode == "minmax":
			min_v = windows.min(axis=1, keepdims=True)
			max_v = windows.max(axis=1, keepdims=True)
			return (windows - min_v) / (max_v - min_v + 1e-8)
		raise ValueError("normalize_mode must be 'zscore' or 'minmax'.")

	def _format_windows(self, windows: np.ndarray) -> np.ndarray:
		data = np.transpose(windows, (0, 2, 1))
		data = data.reshape(data.shape[0], data.shape[1], 1, data.shape[2])
		return data

	def __len__(self) -> int:
		return len(self.labels)

	def __getitem__(self, idx: int):
		return self.data[idx], self.labels[idx]


__all__ = ["portfolio_load_dataset"]
