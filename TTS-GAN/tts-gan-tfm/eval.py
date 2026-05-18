import os
import numpy as np
import torch

from GANModels import Generator
from dataLoader import portfolio_load_dataset, DEFAULT_TICKERS
from visualizationMetrics import JarqueBera, LjungBox, FrobeniusDistance, plot_asset_dashboard

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")

# Activos conocidos para detectar el nombre desde el run
ALL_ASSET_NAMES = list(DEFAULT_TICKERS.keys())


def infer_config_from_state_dict(state_dict):
    """Infiere channels, latent_dim y seq_len desde los pesos del generador."""
    channels  = state_dict["deconv.0.weight"].shape[0]
    latent_dim = state_dict["l1.weight"].shape[1]
    seq_len   = state_dict["pos_embed"].shape[1]
    return channels, latent_dim, seq_len


def detect_asset_from_name(run_name, channels):
    """Intenta detectar el activo desde el nombre del run."""
    name_upper = run_name.upper()
    for asset in ALL_ASSET_NAMES:
        if asset.upper() in name_upper:
            return asset
    # Si hay 1 activo (3 canales con intraday) y no se detecta, asumir UST10Y
    if channels == 3:
        return "UST10Y"
    if channels == 1:
        return "UST10Y"
    return None  # multiples activos


def load_real_data(asset, channels, seq_len, n_samples=1000, filter_regime=["moderate", "stress"]):
    """Carga datos reales del activo correspondiente al modelo."""
    use_intraday = (channels % 3 == 0) and (channels // 3 <= len(ALL_ASSET_NAMES))
    n_assets_model = channels // 3 if use_intraday else channels

    if n_assets_model == 1 and asset is not None:
        assets_list = [asset]
    else:
        assets_list = None  # todos los activos

    dataset = portfolio_load_dataset(
        data_mode="Test",
        assets=assets_list,
        window_length=seq_len,
        stride=1,
        train_ratio=0.8,
        log_returns=True,
        normalize_mode="zscore",
        label_mode="regime",
        filter_regime=filter_regime,
        use_intraday=use_intraday,
    )

    N = min(n_samples, len(dataset))
    real = np.stack([dataset[i][0] for i in range(N)])   # (N, C, 1, T)
    real = np.transpose(real.squeeze(2), (0, 2, 1))       # (N, T, C)
    return real


def eval_run(run_dir):
    ckpt_path = os.path.join(run_dir, "Model", "checkpoint")
    if not os.path.exists(ckpt_path):
        return None

    run_name = os.path.basename(run_dir)
    print(f"\n{'='*60}")
    print(f"Modelo: {run_name}")
    print(f"Checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["avg_gen_state_dict"]

    channels, latent_dim, seq_len = infer_config_from_state_dict(state_dict)
    asset = detect_asset_from_name(run_name, channels)
    print(f"Config inferida → channels={channels}, latent_dim={latent_dim}, seq_len={seq_len}")
    print(f"Activo detectado → {asset if asset else 'todos'}")

    gen = Generator(seq_len=seq_len, patch_size=15, channels=channels, latent_dim=latent_dim)
    gen.load_state_dict(state_dict)
    gen.eval()

    try:
        real = load_real_data(asset, channels, seq_len, filter_regime=["moderate", "stress"])
    except Exception as e:
        print(f"Error cargando datos reales: {e}")
        return None

    N = len(real)
    z = torch.randn(N, latent_dim)
    with torch.no_grad():
        fake = gen(z).cpu().numpy()
    fake = np.transpose(fake.squeeze(2), (0, 2, 1))

    print(f"Datos moderate+stress → reales: {real.shape} | generados: {fake.shape}")

    jb_ori_stat, jb_gen_stat, jb_stat_diff, jb_ori_pval, jb_gen_pval, jb_pval_diff = JarqueBera(real, fake)
    lb_ori, lb_gen, lb_diff = LjungBox(real, fake)
    frob = FrobeniusDistance(real, fake)

    jb_ratio = jb_gen_stat / jb_ori_stat if jb_ori_stat > 0 else float("inf")
    print(f"Jarque-Bera estadístico → Original: {jb_ori_stat:.2f} | Generado: {jb_gen_stat:.2f} | Ratio: {jb_ratio:.2f}x")
    print(f"Jarque-Bera p-valor     → Original: {jb_ori_pval:.2e} | Generado: {jb_gen_pval:.2e}")
    print(f"Ljung-Box²  p-valor     → Original: {lb_ori:.6f} | Generado: {lb_gen:.6f} | Diferencia: {lb_diff:.6f}")
    print(f"Frobenius               → {frob:.6f}")

    metrics = {
        "jb_ori_stat": jb_ori_stat, "jb_gen_stat": jb_gen_stat, "jb_ratio": jb_ratio,
        "jb_ori_pval": jb_ori_pval, "jb_gen_pval": jb_gen_pval,
        "lb_ori": lb_ori, "lb_gen": lb_gen, "lb_diff": lb_diff,
        "frob": frob,
    }
    os.makedirs("images", exist_ok=True)
    safe_name = run_name.replace(":", "-")
    plot_asset_dashboard(real, fake, metrics, asset if asset else "portfolio",
                         f"images/{safe_name}_dashboard.png")

    # Criterios de aceptacion (Tabla 6.8 TFM)
    jb_pass   = jb_gen_pval < 0.05   # rechaza normalidad
    lb_pass   = lb_gen < 0.05        # detecta efecto GARCH
    frob_pass = 0.87 <= frob <= 1.31
    print(f"Criterios → JB {'OK' if jb_pass else 'FALLO'} | "
          f"LB² {'OK' if lb_pass else 'FALLO'} | "
          f"Frobenius {'OK' if frob_pass else 'FALLO'} (rango 0.87–1.31)")

    return {
        "run": run_name,
        "channels": channels,
        "asset": asset if asset else "todos",
        **metrics,
        "jb_pass": jb_pass, "lb_pass": lb_pass, "frob_pass": frob_pass,
    }


def main():
    runs = sorted(
        [d for d in os.listdir(LOGS_DIR) if os.path.isdir(os.path.join(LOGS_DIR, d))]
    )
    if not runs:
        raise FileNotFoundError(f"No hay experimentos en {LOGS_DIR}")

    resultados = []
    for run in runs:
        run_dir = os.path.join(LOGS_DIR, run)
        res = eval_run(run_dir)
        if res is not None:
            resultados.append(res)

    if not resultados:
        print("\nNo se encontro ningun checkpoint valido en logs/")
        return

    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print(f"{'='*60}")
    header = f"{'Run':<35} {'Ch':>3} {'Activo':<10} {'JB_ratio':>9} {'JB_pval':>9} {'LB²_pval':>9} {'Frob':>8}  Criterios"
    print(header)
    print("-" * len(header))
    for r in resultados:
        criterios = f"JB={'OK' if r['jb_pass'] else 'X'} LB={'OK' if r['lb_pass'] else 'X'} Fr={'OK' if r['frob_pass'] else 'X'}"
        print(f"{r['run']:<35} {r['channels']:>3} {r['asset']:<10} "
              f"{r['jb_ratio']:>8.2f}x {r['jb_gen_pval']:>9.2e} {r['lb_gen']:>9.4f} {r['frob']:>8.4f}  {criterios}")


if __name__ == "__main__":
    main()
