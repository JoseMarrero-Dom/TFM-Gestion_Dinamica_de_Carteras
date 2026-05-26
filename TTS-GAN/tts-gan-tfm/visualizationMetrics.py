"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.
Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------
visualization_metrics.py
Note: Use PCA or tSNE for generated and original data visualization
"""

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.stats.diagnostic import acorr_ljungbox

   
def visualization(ori_data, generated_data, save_path, asset_name=""):
    """PCA y t-SNE en una sola figura para comparar la estructura de las
    ventanas reales y sintéticas tras reducir a 2 dimensiones.

    Cada ventana (T pasos × dim canales) se aplana a un vector y luego se
    reducen con PCA y t-SNE. Si las dos nubes de puntos se solapan, la GAN
    captura la geometría global de las series reales aunque proyectada a
    pocas dimensiones.

    Args:
        ori_data:       (N, T, dim) datos reales
        generated_data: (N, T, dim) datos sintéticos
        save_path:      ruta completa del PNG de salida
        asset_name:     nombre opcional para el título
    """
    ori = np.asarray(ori_data)
    gen = np.asarray(generated_data)

    N = min(1000, len(ori), len(gen))
    # RNG local con seed fijo: garantiza que el subconjunto de reales sea
    # idéntico entre runs distintos (si no, t-SNE da embeddings distintos
    # del mismo conjunto solo por el orden de entrada).
    rng = np.random.default_rng(42)
    idx_o = rng.permutation(len(ori))[:N]
    idx_g = rng.permutation(len(gen))[:N]
    ori, gen = ori[idx_o], gen[idx_g]

    # aplanar cada ventana: (N, T*dim) — conserva más info que promediar canales
    flat_ori = ori.reshape(N, -1)
    flat_gen = gen.reshape(N, -1)

    # PCA ajustado solo sobre reales y aplicado a ambos: si la GAN está
    # en otra región del espacio, se ve claramente
    pca = PCA(n_components=2)
    pca.fit(flat_ori)
    pca_o = pca.transform(flat_ori)
    pca_g = pca.transform(flat_gen)

    # t-SNE ajustado sobre la concatenación
    tsne = TSNE(n_components=2, perplexity=min(40, max(5, N // 5)),
                max_iter=500, init="pca", random_state=42)
    tsne_all = tsne.fit_transform(np.concatenate([flat_ori, flat_gen], axis=0))
    tsne_o = tsne_all[:N]
    tsne_g = tsne_all[N:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title = f"Estructura 2D — {asset_name}" if asset_name else "Estructura 2D"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    var_exp = pca.explained_variance_ratio_
    axes[0].scatter(pca_o[:, 0], pca_o[:, 1], c="steelblue", alpha=0.35, s=12, label="Real")
    axes[0].scatter(pca_g[:, 0], pca_g[:, 1], c="tomato",    alpha=0.35, s=12, label="Generado")
    axes[0].set_title(f"PCA  (var explicada: {var_exp[0]*100:.1f}% + {var_exp[1]*100:.1f}%)", fontsize=10)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend(fontsize=9)

    axes[1].scatter(tsne_o[:, 0], tsne_o[:, 1], c="steelblue", alpha=0.35, s=12, label="Real")
    axes[1].scatter(tsne_g[:, 0], tsne_g[:, 1], c="tomato",    alpha=0.35, s=12, label="Generado")
    axes[1].set_title("t-SNE  (perplexity adaptada)", fontsize=10)
    axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"PCA/t-SNE guardado → {save_path}")

def JarqueBera(ori_data, generated_data, alpha=0.05):
    """Prueba de normalidad Jarque-Bera comparada por canal.

    Para cada canal aplica JB sobre los retornos aplanados (todas las ventanas
    concatenadas) y devuelve la fracción de canales que rechazan H0:normalidad.
    Promediar p-valores entre canales no es válido estadísticamente; lo que
    interesa es comparar la tasa de rechazo: un GAN bueno debe reproducir la
    no-normalidad de los retornos financieros (Cont, 2001).

    Returns:
        dict con:
          jb_stat_real / jb_stat_gen   mediana de la estadística JB
          reject_rate_real / _gen      % canales con p<alpha
          reject_rate_diff_pp          diferencia absoluta en puntos porcentuales
    """
    from scipy import stats

    ori = np.asarray(ori_data)
    gen = np.asarray(generated_data)

    N = min(1000, len(ori))
    idx = np.random.permutation(len(ori))[:N]
    ori, gen = ori[idx], gen[idx]
    _, _, dim = ori.shape

    stats_ori, stats_gen = [], []
    pvals_ori, pvals_gen = [], []
    for d in range(dim):
        s_o, p_o = stats.jarque_bera(ori[:, :, d].flatten())
        s_g, p_g = stats.jarque_bera(gen[:, :, d].flatten())
        stats_ori.append(s_o); stats_gen.append(s_g)
        pvals_ori.append(p_o); pvals_gen.append(p_g)

    reject_real = float(np.mean(np.array(pvals_ori) < alpha))
    reject_gen  = float(np.mean(np.array(pvals_gen) < alpha))
    return {
        "jb_stat_real":   float(np.median(stats_ori)),
        "jb_stat_gen":    float(np.median(stats_gen)),
        "reject_real":    reject_real,
        "reject_gen":     reject_gen,
        "reject_diff_pp": abs(reject_real - reject_gen) * 100.0,
    }


def LjungBox(ori_data, generated_data, lags=10, alpha=0.05):
    """Ljung-Box sobre r²: detección de volatility clustering (efecto GARCH).

    Promediar p-valores entre ventanas no es válido. Lo que interesa es la
    tasa de ventanas con autocorrelación significativa en r² — los retornos
    financieros la presentan casi siempre (Cont, 2001). Se calcula también
    Ljung-Box sobre retornos lineales: en mercados eficientes debe ser no
    significativo y un GAN bueno también.

    Returns:
        dict con tasas de rechazo (en r² y en r) para real y generado, y
        diferencia en puntos porcentuales.
    """
    ori = np.asarray(ori_data)
    gen = np.asarray(generated_data)

    N = min(1000, len(ori))
    idx = np.random.permutation(len(ori))[:N]
    ori, gen = ori[idx], gen[idx]
    no, seq_len, dim = ori.shape
    eff_lags = min(lags, seq_len - 1)

    rej_sq_real = rej_sq_gen = 0
    rej_ln_real = rej_ln_gen = 0
    total = no * dim
    for i in range(no):
        for d in range(dim):
            r_o = ori[i, :, d]
            r_g = gen[i, :, d]
            p_sq_o = acorr_ljungbox(r_o ** 2, lags=[eff_lags], return_df=True)['lb_pvalue'].values[-1]
            p_sq_g = acorr_ljungbox(r_g ** 2, lags=[eff_lags], return_df=True)['lb_pvalue'].values[-1]
            p_ln_o = acorr_ljungbox(r_o,       lags=[eff_lags], return_df=True)['lb_pvalue'].values[-1]
            p_ln_g = acorr_ljungbox(r_g,       lags=[eff_lags], return_df=True)['lb_pvalue'].values[-1]
            rej_sq_real += (p_sq_o < alpha); rej_sq_gen += (p_sq_g < alpha)
            rej_ln_real += (p_ln_o < alpha); rej_ln_gen += (p_ln_g < alpha)

    return {
        "garch_real":      rej_sq_real / total,
        "garch_gen":       rej_sq_gen  / total,
        "garch_diff_pp":   abs(rej_sq_real - rej_sq_gen) / total * 100.0,
        "linear_ac_real":  rej_ln_real / total,
        "linear_ac_gen":   rej_ln_gen  / total,
        "linear_ac_diff_pp": abs(rej_ln_real - rej_ln_gen) / total * 100.0,
    }


def FrobeniusDistance(ori_data, generated_data):
    """Distancia de Frobenius relativa entre matrices de correlación.

    El valor absoluto depende del número de canales (la matriz crece con dim),
    así que normalizamos por ||C_real||_F para tener un error relativo
    comparable entre experimentos.

    Returns:
        dict con frob_abs (||ΔC||_F) y frob_rel (||ΔC||_F / ||C_real||_F).
    """
    ori = np.asarray(ori_data)
    gen = np.asarray(generated_data)

    N = min(1000, len(ori))
    idx = np.random.permutation(len(ori))[:N]
    ori, gen = ori[idx], gen[idx]
    _, _, dim = ori.shape

    corr_o = np.corrcoef(ori.reshape(-1, dim), rowvar=False)
    corr_g = np.corrcoef(gen.reshape(-1, dim), rowvar=False)

    # con dim=1 corrcoef devuelve un escalar; lo convertimos en matriz 1x1
    if np.ndim(corr_o) == 0:
        corr_o = np.array([[corr_o]]); corr_g = np.array([[corr_g]])

    frob_abs  = float(np.linalg.norm(corr_o - corr_g, ord='fro'))
    denom     = float(np.linalg.norm(corr_o, ord='fro'))
    frob_rel  = frob_abs / denom if denom > 0 else float("inf")
    return {"frob_abs": frob_abs, "frob_rel": frob_rel}


def plot_asset_dashboard(ori_data, generated_data, metrics, asset_name, save_path):
    """PNG con todas las métricas visuales para un activo.

    Layout:
      Fila 1: histograma real vs generado por canal
      Fila 2: ACF de retornos al cuadrado por canal (efecto GARCH)
      Fila 3: matrices de correlación real y generada + diferencia
      Fila 4: panel de métricas numéricas

    Args:
        ori_data:       (N, T, dim)
        generated_data: (N, T, dim)
        metrics:        dict con JB, LB, Frobenius ya calculados
        asset_name:     str, ej. "UST10Y"
        save_path:      ruta completa del PNG de salida
    """
    from statsmodels.tsa.stattools import acf

    ori_data = np.asarray(ori_data)
    gen_data = np.asarray(generated_data)

    N = min(1000, len(ori_data))
    idx = np.random.permutation(len(ori_data))[:N]
    ori = ori_data[idx]
    gen = gen_data[idx]

    _, T, dim = ori.shape

    # Nombres de canales: logret / oc_range / hl_range por activo
    assets_list = [asset_name] if asset_name else [f"asset{i}" for i in range(dim // 3)]
    suffixes = ["_logret", "_oc_range", "_hl_range"]
    if dim % 3 == 0:
        channel_names = [a + s for a in assets_list for s in suffixes]
    else:
        channel_names = [f"ch{d}" for d in range(dim)]

    n_lags = 20
    n_corr_rows = 1  # fila para heatmaps solo si dim > 1

    total_rows = 3 + n_corr_rows if dim > 1 else 3
    fig = plt.figure(figsize=(max(14, dim * 4), total_rows * 3 + 1.5))
    fig.suptitle(f"TTS-GAN Dashboard — {asset_name}  (moderate + stress)", fontsize=13, fontweight="bold")

    n_cols = max(dim, 3)  # al menos 3 columnas para los heatmaps

    # ── Fila 1: histogramas ───────────────────────────────────────────
    for d in range(dim):
        ax = fig.add_subplot(total_rows, n_cols, d + 1)
        ori_flat = ori[:, :, d].flatten()
        gen_flat = gen[:, :, d].flatten()
        lim = np.percentile(np.abs(np.concatenate([ori_flat, gen_flat])), 99)
        bins = np.linspace(-lim, lim, 60)
        ax.hist(ori_flat, bins=bins, alpha=0.5, color="steelblue", density=True, label="Real")
        ax.hist(gen_flat, bins=bins, alpha=0.5, color="tomato",    density=True, label="Generado")
        ax.set_title(channel_names[d], fontsize=8)
        ax.set_xlabel("valor normalizado", fontsize=7)
        if d == 0:
            ax.legend(fontsize=7)
            ax.set_ylabel("densidad", fontsize=7)
        ax.tick_params(labelsize=7)

    # ── Fila 2: ACF de retornos al cuadrado (GARCH) ───────────────────
    for d in range(dim):
        ax = fig.add_subplot(total_rows, n_cols, n_cols + d + 1)
        ori_sq = ori[:, :, d].flatten() ** 2
        gen_sq = gen[:, :, d].flatten() ** 2
        acf_ori = acf(ori_sq, nlags=n_lags, fft=True)
        acf_gen = acf(gen_sq, nlags=n_lags, fft=True)
        lags = np.arange(n_lags + 1)
        ax.bar(lags - 0.2, acf_ori, width=0.35, color="steelblue", alpha=0.7, label="Real")
        ax.bar(lags + 0.2, acf_gen, width=0.35, color="tomato",    alpha=0.7, label="Gen")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline(1.96 / np.sqrt(len(ori_sq)), color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"ACF r² — {channel_names[d]}", fontsize=8)
        ax.set_xlabel("lag", fontsize=7)
        if d == 0:
            ax.legend(fontsize=7)
            ax.set_ylabel("ACF", fontsize=7)
        ax.tick_params(labelsize=7)

    # ── Fila 3: matrices de correlación (solo si dim > 1) ─────────────
    if dim > 1:
        ori_flat_all = ori.reshape(-1, dim)
        gen_flat_all = gen.reshape(-1, dim)
        corr_ori = np.corrcoef(ori_flat_all, rowvar=False)
        corr_gen = np.corrcoef(gen_flat_all, rowvar=False)
        corr_diff = corr_ori - corr_gen

        for col_i, (mat, title) in enumerate([(corr_ori, "Corr Real"),
                                               (corr_gen, "Corr Generado"),
                                               (corr_diff, "Diferencia")]):
            ax = fig.add_subplot(total_rows, n_cols, 2 * n_cols + col_i + 1)
            vmax = 1.0 if col_i < 2 else np.abs(corr_diff).max()
            vmin = -1.0 if col_i < 2 else -vmax
            im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="auto")
            ax.set_title(title, fontsize=8)
            ax.set_xticks(range(dim))
            ax.set_yticks(range(dim))
            short = [c.replace(asset_name + "_", "") for c in channel_names]
            ax.set_xticklabels(short, rotation=45, fontsize=6)
            ax.set_yticklabels(short, fontsize=6)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        row_metrics = 3
    else:
        row_metrics = 2

    # ── Fila final: panel numérico ────────────────────────────────────
    ax_txt = fig.add_subplot(total_rows, 1, total_rows)
    ax_txt.axis("off")

    jb       = metrics["jb"]
    lb       = metrics["lb"]
    frob     = metrics["frob"]
    moments  = metrics["moments"]
    acf_lin  = metrics["acf_linear"]

    # criterios justificados por la literatura de hechos estilizados (Cont, 2001):
    #   - los retornos rechazan normalidad casi siempre → tasas similares
    #   - presentan volatility clustering (Ljung-Box r² significativo)
    #   - ACF lineal ≈ 0 (mercado eficiente)
    #   - correlaciones contemporáneas estables → Frobenius relativo pequeño
    jb_pass   = jb["reject_diff_pp"] < 10.0
    lb_pass   = lb["garch_diff_pp"]  < 15.0 and lb["garch_gen"] > 0.5
    frob_pass = frob["frob_rel"] < 0.30
    acf_pass  = acf_lin["acf_diff_med"] < 0.05
    kurt_pass = moments["kurt_mean_rel_err"] < 0.25

    def ok(v):
        return "✓ OK" if v else "✗ FALLO"

    lines = [
        f"Jarque-Bera   → rechazo H0 (normalidad)  Real: {jb['reject_real']*100:5.1f}%  Gen: {jb['reject_gen']*100:5.1f}%  Δ={jb['reject_diff_pp']:.1f}pp   {ok(jb_pass)}  (Δ<10pp)",
        f"Ljung-Box r²  → GARCH presente            Real: {lb['garch_real']*100:5.1f}%  Gen: {lb['garch_gen']*100:5.1f}%  Δ={lb['garch_diff_pp']:.1f}pp   {ok(lb_pass)}  (Δ<15pp y gen>50%)",
        f"ACF lineal    → mediana |ACF(1)|          Real: {acf_lin['acf_real_med']:.3f}    Gen: {acf_lin['acf_gen_med']:.3f}    Δ={acf_lin['acf_diff_med']:.3f}   {ok(acf_pass)}  (Δ<0.05)",
        f"Frobenius     → ||ΔC||_F relativo:        {frob['frob_rel']:.3f}   {ok(frob_pass)}  (<0.30)",
        f"Momentos      → skew |Δ|={moments['skew_mean_abs_diff']:.3f}   kurt err_rel={moments['kurt_mean_rel_err']:.3f}   {ok(kurt_pass)}  (kurt<0.25)",
    ]
    ax_txt.text(0.5, 0.5, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=9, va="center", ha="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                family="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard guardado → {save_path}")


def MomentsComparison(ori_data, generated_data):
    """Momentos por canal: media, std, skewness, kurtosis.

    Aplanar todos los canales mezcla activos con escalas y signos distintos
    (logret vs ranges) y oculta dónde falla el GAN. Como los datos vienen
    z-score-normalizados, media≈0 y std≈1 en real por construcción, así que
    el peso real lo cargan skewness y kurtosis (colas).

    Returns:
        dict con momentos por canal y diferencia absoluta promedio en skew/kurt.
    """
    from scipy import stats as sp_stats

    ori = np.asarray(ori_data)
    gen = np.asarray(generated_data)
    _, _, dim = ori.shape

    per_channel = []
    for d in range(dim):
        o = ori[:, :, d].flatten()
        g = gen[:, :, d].flatten()
        per_channel.append({
            "mean_real":  float(np.mean(o)),     "mean_gen":  float(np.mean(g)),
            "std_real":   float(np.std(o)),      "std_gen":   float(np.std(g)),
            "skew_real":  float(sp_stats.skew(o)),     "skew_gen":  float(sp_stats.skew(g)),
            "kurt_real":  float(sp_stats.kurtosis(o)), "kurt_gen":  float(sp_stats.kurtosis(g)),
        })

    skew_diff = np.mean([abs(c["skew_real"] - c["skew_gen"]) for c in per_channel])
    # kurt: usamos error relativo porque suele variar mucho entre canales
    def _rel(a, b):
        denom = abs(a) if abs(a) > 1e-6 else 1.0
        return abs(a - b) / denom
    kurt_rel_err = np.mean([_rel(c["kurt_real"], c["kurt_gen"]) for c in per_channel])

    return {
        "per_channel": per_channel,
        "skew_mean_abs_diff": float(skew_diff),
        "kurt_mean_rel_err":  float(kurt_rel_err),
    }


def VaRCVaR(ori_data, generated_data, levels=(0.95, 0.99), logret_channels=None):
    """VaR y CVaR por canal de logret a 95% y 99%.

    Aplanar todos los canales mezcla logrets con ranges OC/HL (que viven en
    escalas y signos distintos) y rompe la interpretación financiera de la
    cola. Si se pasa logret_channels=lista_índices, solo se evalúan esos
    canales (típicamente los de retornos), si no, se evalúan todos.

    Returns:
        dict con error relativo promedio (sobre canales) en VaR y CVaR a 95%.
    """
    ori = np.asarray(ori_data)
    gen = np.asarray(generated_data)
    _, _, dim = ori.shape
    chans = logret_channels if logret_channels is not None else list(range(dim))

    per_level = {}
    for alpha in levels:
        q = 1.0 - alpha
        var_rel_errs, cvar_rel_errs = [], []
        per_chan = []
        for d in chans:
            o = ori[:, :, d].flatten()
            g = gen[:, :, d].flatten()
            var_o = float(np.quantile(o, q));  var_g = float(np.quantile(g, q))
            tail_o = o[o <= var_o];            tail_g = g[g <= var_g]
            cvar_o = float(tail_o.mean()) if len(tail_o) else var_o
            cvar_g = float(tail_g.mean()) if len(tail_g) else var_g
            denom_v = abs(var_o)  if abs(var_o)  > 1e-6 else 1.0
            denom_c = abs(cvar_o) if abs(cvar_o) > 1e-6 else 1.0
            var_rel_errs.append(abs(var_o - var_g) / denom_v)
            cvar_rel_errs.append(abs(cvar_o - cvar_g) / denom_c)
            per_chan.append({
                "channel": d,
                "VaR_real": var_o,  "VaR_gen": var_g,
                "CVaR_real": cvar_o, "CVaR_gen": cvar_g,
            })
        per_level[alpha] = {
            "per_channel":      per_chan,
            "var_mean_rel_err": float(np.mean(var_rel_errs)),
            "cvar_mean_rel_err":float(np.mean(cvar_rel_errs)),
        }
    return per_level


def ACFLinear(ori_data, generated_data, lag=1):
    """ACF de retornos lineales a lag dado.

    Por hipótesis de mercado eficiente, la ACF de los retornos (no del
    cuadrado) debe estar cerca de 0. Si el GAN produce ACF lineal alta,
    está introduciendo previsibilidad espuria. Evaluado por canal.

    Returns:
        dict con acf real/gen medianos en valor absoluto y diferencia.
    """
    from statsmodels.tsa.stattools import acf

    ori = np.asarray(ori_data)
    gen = np.asarray(generated_data)
    _, _, dim = ori.shape

    acfs_real, acfs_gen = [], []
    for d in range(dim):
        a_o = acf(ori[:, :, d].flatten(), nlags=lag, fft=True)[lag]
        a_g = acf(gen[:, :, d].flatten(), nlags=lag, fft=True)[lag]
        acfs_real.append(abs(a_o))
        acfs_gen.append(abs(a_g))

    return {
        "acf_lag":      lag,
        "acf_real_med": float(np.median(acfs_real)),
        "acf_gen_med":  float(np.median(acfs_gen)),
        "acf_diff_med": float(abs(np.median(acfs_real) - np.median(acfs_gen))),
    }
