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
