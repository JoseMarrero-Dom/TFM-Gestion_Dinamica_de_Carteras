import os
import numpy as np
import torch

from GANModels import Generator
from dataLoader import portfolio_load_dataset, DEFAULT_TICKERS
from visualizationMetrics import (
    JarqueBera, LjungBox, FrobeniusDistance,
    MomentsComparison, VaRCVaR, ACFLinear,
    plot_asset_dashboard, visualization,
)

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")

# Activos conocidos para detectar el nombre desde el run
ALL_ASSET_NAMES = list(DEFAULT_TICKERS.keys())


def infer_config_from_state_dict(state_dict):
    """Infiere channels, latent_dim y seq_len desde los pesos del generador."""
    channels  = state_dict["deconv.0.weight"].shape[0]
    latent_dim = state_dict["l1.weight"].shape[1]
    seq_len   = state_dict["pos_embed"].shape[1]
    return channels, latent_dim, seq_len


def detect_assets_from_name(run_name, channels):
    """Devuelve lista de activos inferida del nombre del run y los canales.

    - channels == 18 y 'portfolio' en el nombre → todos los activos
    - channels == 18 en cualquier caso → todos los activos (modelo conjunto)
    - channels == 3  → un solo activo; lo busca en el nombre o asume UST10Y
    - channels == 1  → idem sin intraday
    """
    if channels == 18:
        return None   # None = todos los activos en dataLoader

    # modelo mono-activo
    name_upper = run_name.upper()
    for asset in ALL_ASSET_NAMES:
        if asset.upper() in name_upper:
            return [asset]
    return ["UST10Y"]  # fallback


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
        ckpt_path = os.path.join(run_dir, "Model", "checkpoint.zip")
    if not os.path.exists(ckpt_path):
        return None

    run_name = os.path.basename(run_dir)
    print(f"\n{'='*60}")
    print(f"Modelo: {run_name}")
    print(f"Checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["avg_gen_state_dict"]

    channels, latent_dim, seq_len = infer_config_from_state_dict(state_dict)
    assets = detect_assets_from_name(run_name, channels)
    asset  = assets[0] if assets and len(assets) == 1 else None
    print(f"Config inferida → channels={channels}, latent_dim={latent_dim}, seq_len={seq_len}")
    print(f"Activos detectados → {assets if assets else 'todos'}")

    gen = Generator(seq_len=seq_len, patch_size=15, channels=channels, latent_dim=latent_dim)
    gen.load_state_dict(state_dict)
    gen.eval()

    try:
        real = load_real_data(asset, channels, seq_len, filter_regime=["moderate", "stress"])
    except Exception as e:
        print(f"Error cargando datos reales: {e}")
        return None

    # seed fijo para reproducibilidad: si no se fija aquí, las tasas de rechazo
    # JB/LB, kurtosis y demás varían entre ejecuciones porque gen(torch.randn(...))
    # ve ruido distinto cada vez.
    SAMPLING_SEED = 42
    torch.manual_seed(SAMPLING_SEED)
    np.random.seed(SAMPLING_SEED)

    N = len(real)
    z = torch.randn(N, latent_dim)
    with torch.no_grad():
        fake = gen(z).cpu().numpy()
    fake = np.transpose(fake.squeeze(2), (0, 2, 1))

    print(f"Datos moderate+stress → reales: {real.shape} | generados: {fake.shape}")

    # canales de logret: si la GAN usa 3 canales/activo (logret, oc, hl), los logret son 0,3,6,...
    use_intraday  = (channels % 3 == 0)
    n_assets_eval = channels // 3 if use_intraday else channels
    logret_chans  = list(range(0, channels, 3)) if use_intraday else list(range(channels))

    jb_global   = JarqueBera(real, fake)
    lb_global   = LjungBox(real, fake)
    frob_global = FrobeniusDistance(real, fake)
    mom_global  = MomentsComparison(real, fake)
    var_global  = VaRCVaR(real, fake, logret_channels=logret_chans)
    acf_global  = ACFLinear(real, fake, lag=1)

    print(f"Jarque-Bera → rechazo H0  Real: {jb_global['reject_real']*100:.1f}%  Gen: {jb_global['reject_gen']*100:.1f}%  Δ={jb_global['reject_diff_pp']:.1f}pp")
    print(f"Ljung-Box r² → GARCH      Real: {lb_global['garch_real']*100:.1f}%  Gen: {lb_global['garch_gen']*100:.1f}%  Δ={lb_global['garch_diff_pp']:.1f}pp")
    print(f"Ljung-Box r  → AC lineal  Real: {lb_global['linear_ac_real']*100:.1f}%  Gen: {lb_global['linear_ac_gen']*100:.1f}%  Δ={lb_global['linear_ac_diff_pp']:.1f}pp")
    print(f"ACF(1) lineal mediana     Real: {acf_global['acf_real_med']:.3f}  Gen: {acf_global['acf_gen_med']:.3f}  Δ={acf_global['acf_diff_med']:.3f}")
    print(f"Frobenius relativo        {frob_global['frob_rel']:.3f}  (abs {frob_global['frob_abs']:.3f})")
    print(f"Momentos     skew |Δ|={mom_global['skew_mean_abs_diff']:.3f}   kurt err_rel={mom_global['kurt_mean_rel_err']:.3f}")
    print(f"VaR 95% err_rel={var_global[0.95]['var_mean_rel_err']:.3f}   CVaR 95% err_rel={var_global[0.95]['cvar_mean_rel_err']:.3f}")

    metrics = {
        "jb":         jb_global,
        "lb":         lb_global,
        "frob":       frob_global,
        "moments":    mom_global,
        "var_cvar":   var_global,
        "acf_linear": acf_global,
    }
    os.makedirs("images", exist_ok=True)
    safe_name = run_name.replace(":", "-")

    if channels == 18:
        # un dashboard + un PCA/t-SNE por activo (3 canales: logret, oc_range, hl_range)
        for i, aname in enumerate(ALL_ASSET_NAMES):
            c0, c1 = i * 3, i * 3 + 3
            real_a = real[:, :, c0:c1]
            fake_a = fake[:, :, c0:c1]
            metrics_a = {
                "jb":         JarqueBera(real_a, fake_a),
                "lb":         LjungBox(real_a, fake_a),
                "frob":       FrobeniusDistance(real_a, fake_a),
                "moments":    MomentsComparison(real_a, fake_a),
                "var_cvar":   VaRCVaR(real_a, fake_a, logret_channels=[0]),
                "acf_linear": ACFLinear(real_a, fake_a),
            }
            plot_asset_dashboard(real_a, fake_a, metrics_a, aname,
                                 f"images/{safe_name}_{aname}_dashboard.png")
            visualization(real_a, fake_a,
                          f"images/{safe_name}_{aname}_pca_tsne.png", asset_name=aname)
        # un PCA/t-SNE conjunto con todos los canales (vista global del portfolio)
        visualization(real, fake,
                      f"images/{safe_name}_portfolio_pca_tsne.png", asset_name="portfolio")
    else:
        label = asset if asset else "portfolio"
        plot_asset_dashboard(real, fake, metrics, label,
                             f"images/{safe_name}_dashboard.png")
        visualization(real, fake,
                      f"images/{safe_name}_pca_tsne.png", asset_name=label)

    # Criterios (hechos estilizados de Cont, 2001):
    #   JB: tasas de rechazo de normalidad parecidas en real y gen (Δ<10pp).
    #   LB r²: GARCH presente en gen y diferencia con real pequeña.
    #   LB r y ACF(1) lineal: cerca de 0 en ambos (mercado eficiente).
    #   Frobenius relativo: estructura de correlaciones bien reproducida.
    #   Kurtosis: error relativo < 25% (las colas son críticas para riesgo).
    jb_pass   = jb_global["reject_diff_pp"] < 10.0
    lb_pass   = lb_global["garch_diff_pp"]  < 15.0 and lb_global["garch_gen"] > 0.5
    acf_pass  = acf_global["acf_diff_med"]  < 0.05
    frob_pass = frob_global["frob_rel"]      < 0.30
    kurt_pass = mom_global["kurt_mean_rel_err"] < 0.25
    print(f"Criterios → JB {'OK' if jb_pass else 'FALLO'} | "
          f"LB² {'OK' if lb_pass else 'FALLO'} | "
          f"ACF {'OK' if acf_pass else 'FALLO'} | "
          f"Frob {'OK' if frob_pass else 'FALLO'} | "
          f"Kurt {'OK' if kurt_pass else 'FALLO'}")

    return {
        "run": run_name,
        "channels": channels,
        "asset": asset if asset else "todos",
        "jb_reject_real_pct":  jb_global["reject_real"] * 100,
        "jb_reject_gen_pct":   jb_global["reject_gen"]  * 100,
        "jb_reject_diff_pp":   jb_global["reject_diff_pp"],
        "lb_garch_real_pct":   lb_global["garch_real"] * 100,
        "lb_garch_gen_pct":    lb_global["garch_gen"]  * 100,
        "lb_garch_diff_pp":    lb_global["garch_diff_pp"],
        "acf_real":            acf_global["acf_real_med"],
        "acf_gen":             acf_global["acf_gen_med"],
        "frob_rel":            frob_global["frob_rel"],
        "kurt_err":            mom_global["kurt_mean_rel_err"],
        "var95_err":           var_global[0.95]["var_mean_rel_err"],
        "cvar95_err":          var_global[0.95]["cvar_mean_rel_err"],
        "jb_pass": jb_pass, "lb_pass": lb_pass, "acf_pass": acf_pass,
        "frob_pass": frob_pass, "kurt_pass": kurt_pass,
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

    print(f"\n{'='*100}")
    print("RESUMEN FINAL")
    print(f"{'='*100}")
    header = (f"{'Run':<35} {'Ch':>3} {'Activo':<10} "
              f"{'JB_Δpp':>7} {'GARCH_Δpp':>10} {'ACF_Δ':>7} {'Frob_rel':>9} {'Kurt_err':>9}  Criterios")
    print(header)
    print("-" * len(header))
    for r in resultados:
        crit = (f"JB={'O' if r['jb_pass'] else 'X'} "
                f"LB={'O' if r['lb_pass'] else 'X'} "
                f"AC={'O' if r['acf_pass'] else 'X'} "
                f"Fr={'O' if r['frob_pass'] else 'X'} "
                f"Kt={'O' if r['kurt_pass'] else 'X'}")
        acf_diff = abs(r["acf_real"] - r["acf_gen"])
        print(f"{r['run']:<35} {r['channels']:>3} {r['asset']:<10} "
              f"{r['jb_reject_diff_pp']:>7.1f} {r['lb_garch_diff_pp']:>10.1f} "
              f"{acf_diff:>7.3f} {r['frob_rel']:>9.3f} {r['kurt_err']:>9.3f}  {crit}")


if __name__ == "__main__":
    main()
