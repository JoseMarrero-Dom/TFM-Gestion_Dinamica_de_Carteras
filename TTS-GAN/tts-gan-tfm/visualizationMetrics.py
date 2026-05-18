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

   
def visualization (ori_data, generated_data, analysis, save_name):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
    """  
    # Analysis sample size (for faster computation)
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)  

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape  

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                        np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, 
                                        np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
    
    # Visualization parameter        
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components = 2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)    
        plt.scatter(pca_results[:,0], pca_results[:,1],
                    c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                    c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")

        ax.legend()  
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
#        plt.show()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together       
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)

        # TSNE anlaysis
        tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, max_iter = 300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                    c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
        plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                    c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")

        ax.legend()

        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
#        plt.show()    
        
    plt.savefig(f'./images/{save_name}.png', format="png")
    plt.show()

def JarqueBera(ori_data, generated_data):
    """Prueba de normalidad Jarque-Bera comparando datos originales y generados.

    Compara los p-values de la prueba JB aplicada a cada dimensión,
    promediando sobre todas las features y muestras.

    Args:
        ori_data:       datos originales,   shape (N, seq_len, dim)
        generated_data: datos sintéticos,   shape (N, seq_len, dim)

    Returns:
        jb_ori_mean:  media de p-values JB sobre datos originales
        jb_gen_mean:  media de p-values JB sobre datos generados
        jb_diff:      diferencia absoluta entre ambas medias
    """
    from scipy import stats

    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    anal_sample_no = min(1000, len(ori_data))
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    jb_ori_stats  = []
    jb_gen_stats  = []
    jb_ori_pvalues = []
    jb_gen_pvalues = []

    for d in range(dim):
        ori_flat = ori_data[:, :, d].flatten()
        gen_flat = generated_data[:, :, d].flatten()

        stat_ori, p_ori = stats.jarque_bera(ori_flat)
        stat_gen, p_gen = stats.jarque_bera(gen_flat)

        jb_ori_stats.append(stat_ori)
        jb_gen_stats.append(stat_gen)
        jb_ori_pvalues.append(p_ori)
        jb_gen_pvalues.append(p_gen)

    jb_ori_stat  = np.mean(jb_ori_stats)
    jb_gen_stat  = np.mean(jb_gen_stats)
    jb_ori_pval  = np.mean(jb_ori_pvalues)
    jb_gen_pval  = np.mean(jb_gen_pvalues)
    jb_stat_diff = np.abs(jb_ori_stat - jb_gen_stat)
    jb_pval_diff = np.abs(jb_ori_pval  - jb_gen_pval)

    return jb_ori_stat, jb_gen_stat, jb_stat_diff, jb_ori_pval, jb_gen_pval, jb_pval_diff


def LjungBox(ori_data, generated_data, lags=10):
    """Prueba de autocorrelación Ljung-Box comparando datos originales y generados.

    Evalúa si las series temporales presentan autocorrelación significativa.
    Un buen modelo generativo debería reproducir la estructura de autocorrelación
    del proceso original.

    Args:
        ori_data:       datos originales,   shape (N, seq_len, dim)
        generated_data: datos sintéticos,   shape (N, seq_len, dim)
        lags:           número de retardos a evaluar (default: 10)

    Returns:
        lb_ori_mean:  media de p-values LB sobre datos originales
        lb_gen_mean:  media de p-values LB sobre datos generados
        lb_diff:      diferencia absoluta entre ambas medias
    """

    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    anal_sample_no = min(1000, len(ori_data))
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    lb_ori_pvalues = []
    lb_gen_pvalues = []

    effective_lags = min(lags, seq_len - 1)

    for i in range(no):
        for d in range(dim):
            ori_series = ori_data[i, :, d]
            gen_series = generated_data[i, :, d]

            lb_ori = acorr_ljungbox(ori_series ** 2, lags=[effective_lags], return_df=True)
            lb_gen = acorr_ljungbox(gen_series ** 2, lags=[effective_lags], return_df=True)

            lb_ori_pvalues.append(lb_ori['lb_pvalue'].values[-1])
            lb_gen_pvalues.append(lb_gen['lb_pvalue'].values[-1])

    lb_ori_mean = np.mean(lb_ori_pvalues)
    lb_gen_mean = np.mean(lb_gen_pvalues)
    lb_diff     = np.abs(lb_ori_mean - lb_gen_mean)

    return lb_ori_mean, lb_gen_mean, lb_diff


def FrobeniusDistance(ori_data, generated_data):
    """Distancia de Frobenius entre las matrices de covarianza de los datos
    originales y generados.

    Mide qué tan bien el modelo generativo reproduce la estructura de
    covarianza (correlaciones lineales entre features) del proceso real.

    Args:
        ori_data:       datos originales,   shape (N, seq_len, dim)
        generated_data: datos sintéticos,   shape (N, seq_len, dim)

    Returns:
        frob_dist: distancia de Frobenius entre matrices de covarianza
    """
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    anal_sample_no = min(1000, len(ori_data))
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    ori_flat = ori_data.reshape(-1, dim)
    gen_flat = generated_data.reshape(-1, dim)

    corr_ori = np.corrcoef(ori_flat, rowvar=False)   # (dim, dim)
    corr_gen = np.corrcoef(gen_flat, rowvar=False)   # (dim, dim)

    diff = corr_ori - corr_gen
    frob_dist = np.linalg.norm(diff, ord='fro')

    return frob_dist


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

    jb_pass   = metrics["jb_gen_pval"] < 0.05
    lb_pass   = metrics["lb_gen"] < 0.05
    frob_pass = 0.87 <= metrics["frob"] <= 1.31

    def ok(v):
        return "✓ OK" if v else "✗ FALLO"

    lines = [
        f"Jarque-Bera  estadístico → Real: {metrics['jb_ori_stat']:.1f}  |  "
        f"Generado: {metrics['jb_gen_stat']:.1f}  |  Ratio: {metrics['jb_ratio']:.2f}x   {ok(jb_pass)}",
        f"Jarque-Bera  p-valor     → Real: {metrics['jb_ori_pval']:.2e}  |  Generado: {metrics['jb_gen_pval']:.2e}",
        f"Ljung-Box²   p-valor     → Real: {metrics['lb_ori']:.4f}  |  "
        f"Generado: {metrics['lb_gen']:.4f}  |  Diff: {metrics['lb_diff']:.4f}   {ok(lb_pass)}",
        f"Frobenius    distancia   → {metrics['frob']:.4f}   {ok(frob_pass)}  (criterio 0.87–1.31)",
    ]
    ax_txt.text(0.5, 0.5, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=9, va="center", ha="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                family="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Dashboard guardado → {save_path}")
