#!/usr/bin/env python3
"""
Discriminative Score para evaluar calidad de datos sinteticos del TTS-GAN.

Entrena un LSTM de dos capas para clasificar ventanas como reales o sinteticas.
  - accuracy ~ 0.5  →  GAN bueno  (el clasificador no distingue)
  - accuracy ~ 1.0  →  GAN malo   (los datos son distinguibles)

disc_score = |accuracy - 0.5|  →  cuanto mas cerca de 0, mejor.

Uso:
    python discriminative_score.py --run UST10Y_2026_05_18_18_39_50
    python discriminative_score.py           # evalua todos los runs en logs/
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from GANModels import Generator
from dataLoader import portfolio_load_dataset, DEFAULT_TICKERS

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
ALL_ASSET_NAMES = list(DEFAULT_TICKERS.keys())


# ── Modelo clasificador ────────────────────────────────────────────────────────

class _DiscLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)


# ── Métrica ────────────────────────────────────────────────────────────────────

def discriminative_score(real, fake, epochs=50, hidden_dim=64, seed=42):
    """Calcula el discriminative score entre datos reales y sinteticos.

    Args:
        real:  np.ndarray (N, T, dim)
        fake:  np.ndarray (N, T, dim)

    Returns:
        disc_score: float en [0, 0.5]  — cuanto mas bajo mejor
        accuracy:   float en [0.5, 1]  — accuracy del clasificador en test
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    real = np.asarray(real, dtype=np.float32)
    fake = np.asarray(fake, dtype=np.float32)

    N = min(len(real), len(fake))
    real, fake = real[:N], fake[:N]

    X = np.concatenate([real, fake], axis=0)
    y = np.concatenate([np.ones(N), np.zeros(N)])

    idx = np.random.permutation(2 * N)
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=64, shuffle=True,
    )

    model     = _DiscLSTM(input_dim=X.shape[2], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds    = (torch.sigmoid(model(torch.tensor(X_te))) > 0.5).float()
        accuracy = (preds == torch.tensor(y_te, dtype=torch.float32)).float().mean().item()

    score = abs(accuracy - 0.5)
    return score, accuracy


# ── Carga de datos (igual que en eval.py) ─────────────────────────────────────

def _infer_config(state_dict):
    channels   = state_dict["deconv.0.weight"].shape[0]
    latent_dim = state_dict["l1.weight"].shape[1]
    seq_len    = state_dict["pos_embed"].shape[1]
    return channels, latent_dim, seq_len


def _detect_asset(run_name, channels):
    if channels == 18:
        return None
    name_upper = run_name.upper()
    for asset in ALL_ASSET_NAMES:
        if asset.upper() in name_upper:
            return asset
    return "UST10Y"


def _load_real(asset, channels, seq_len, n=1000):
    use_intraday = (channels % 3 == 0)
    assets_list  = [asset] if asset else None
    dataset = portfolio_load_dataset(
        data_mode="Test",
        assets=assets_list,
        window_length=seq_len,
        stride=1,
        train_ratio=0.8,
        log_returns=True,
        normalize_mode="zscore",
        label_mode="regime",
        filter_regime=["moderate", "stress"],
        use_intraday=use_intraday,
    )
    N    = min(n, len(dataset))
    real = np.stack([dataset[i][0] for i in range(N)])
    real = np.transpose(real.squeeze(2), (0, 2, 1))
    return real


# ── Evaluacion por run ─────────────────────────────────────────────────────────

def eval_run(run_dir):
    ckpt_path = os.path.join(run_dir, "Model", "checkpoint")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(run_dir, "Model", "checkpoint.zip")
    if not os.path.exists(ckpt_path):
        return None

    run_name = os.path.basename(run_dir)
    print(f"\n{'='*55}")
    print(f"Run: {run_name}")

    ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["avg_gen_state_dict"]
    channels, latent_dim, seq_len = _infer_config(state_dict)
    asset = _detect_asset(run_name, channels)

    gen = Generator(seq_len=seq_len, patch_size=15, channels=channels, latent_dim=latent_dim)
    gen.load_state_dict(state_dict)
    gen.eval()

    try:
        real = _load_real(asset, channels, seq_len)
    except Exception as e:
        print(f"Error cargando datos reales: {e}")
        return None

    # seed también en el muestreo: si solo se fija dentro de discriminative_score(),
    # cada lanzamiento genera ventanas fake distintas y el score varía.
    SAMPLING_SEED = 42
    torch.manual_seed(SAMPLING_SEED)
    np.random.seed(SAMPLING_SEED)

    N = len(real)
    with torch.no_grad():
        fake = gen(torch.randn(N, latent_dim)).cpu().numpy()
    fake = np.transpose(fake.squeeze(2), (0, 2, 1))

    score, acc = discriminative_score(real, fake)
    label = "OK" if score < 0.1 else "FALLO"
    print(f"Activo: {asset or 'todos'}  |  accuracy={acc:.4f}  |  disc_score={score:.4f}  [{label}]")
    return {"run": run_name, "asset": asset or "todos", "accuracy": acc, "disc_score": score}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None,
                        help="Nombre del run en logs/ (omitir para evaluar todos)")
    args = parser.parse_args()

    if args.run:
        runs = [args.run]
    else:
        runs = sorted(d for d in os.listdir(LOGS_DIR)
                      if os.path.isdir(os.path.join(LOGS_DIR, d)))

    resultados = []
    for run in runs:
        res = eval_run(os.path.join(LOGS_DIR, run))
        if res:
            resultados.append(res)

    if not resultados:
        print("No se encontro ningun checkpoint valido en logs/")
        return

    print(f"\n{'='*55}")
    print(f"{'Run':<35} {'Activo':<10} {'Accuracy':>9} {'Score':>7}  Criterio")
    print("-" * 65)
    for r in resultados:
        label = "OK" if r["disc_score"] < 0.1 else "FALLO"
        print(f"{r['run']:<35} {r['asset']:<10} {r['accuracy']:>9.4f} {r['disc_score']:>7.4f}  {label}")


if __name__ == "__main__":
    main()
