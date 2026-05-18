import os
import numpy as np
import torch

from GANModels import Generator
from dataLoader import portfolio_load_dataset
from visualizationMetrics import visualization, JarqueBera, LjungBox, FrobeniusDistance

# 1) Cargar checkpoint (el mas reciente de logs/)
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
runs = sorted(
    [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))],
    reverse=True
)
if not runs:
    raise FileNotFoundError(f"No hay experimentos en {logs_dir}")
ckpt_path = os.path.join(logs_dir, runs[0], "Model", "checkpoint")
print(f"Cargando checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# 2) Instanciar generador con los mismos parametros del entrenamiento
gen = Generator(seq_len=150, patch_size=15, channels=3, latent_dim=256)
gen.load_state_dict(ckpt["avg_gen_state_dict"])  # o "gen_state_dict"
gen.eval()

# 3) Datos reales (test o train)
real_set = portfolio_load_dataset(
    data_mode="Test",  # o "Train"
    assets=["UST10Y"],
    window_length=150,
    stride=1,
    train_ratio=0.8,
    log_returns=True,
    normalize_mode="zscore",
    label_mode="regime",
    filter_regime=["moderate", "stress"],
    use_intraday=True,
)

N = min(1000, len(real_set))
real = np.stack([real_set[i][0] for i in range(N)])   # (N, C, 1, T)
real = np.transpose(real.squeeze(2), (0, 2, 1))       # (N, T, C)

# 4) Datos sinteticos
z = torch.randn(N, 256)  # (N, latent_dim)
with torch.no_grad():
    fake = gen(z).cpu().numpy()                        # (N, C, 1, T)
fake = np.transpose(fake.squeeze(2), (0, 2, 1))        # (N, T, C)

# 5) Visualizacion (PCA o tSNE)
os.makedirs("images", exist_ok=True)
visualization(real, fake, analysis="pca", save_name="ust10y_pca")
visualization(real, fake, analysis="tsne", save_name="ust10y_tsne")

# 6) Jarque-Bera
jb_ori_mean, jb_gen_mean, jb_diff = JarqueBera(real, fake)
print(f"Jarque-Bera - Original: {jb_ori_mean}, Generated: {jb_gen_mean}, Difference: {jb_diff}")

# 7) Ljung-Box
lb_ori_mean, lb_gen_mean, lb_diff = LjungBox(real, fake)
print(f"Ljung-Box - Original: {lb_ori_mean}, Generated: {lb_gen_mean}, Difference: {lb_diff}")

# 8) Frobenius Distance
frob_dist = FrobeniusDistance(real, fake)
print(f"Frobenius Distance: {frob_dist}")
