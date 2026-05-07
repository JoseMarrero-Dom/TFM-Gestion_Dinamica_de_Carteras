import os
import numpy as np
import torch

from GANModels import Generator
from dataLoader import portfolio_load_dataset
from visualizationMetrics import visualization

# 1) Cargar checkpoint
ckpt_path = "/root/Home/UNIR/TFM/tfm/TTS-GAN/tts-gan-tfm/logs/Running_2026_05_07_19_32_14/Model/checkpoint"
ckpt = torch.load(ckpt_path, map_location="cpu")

# 2) Instanciar generador con los mismos parametros del entrenamiento
gen = Generator(seq_len=150, patch_size=15, channels=1, latent_dim=100)
gen.load_state_dict(ckpt["avg_gen_state_dict"])  # o "gen_state_dict"
gen.eval()

# 3) Datos reales (test o train)
real_set = portfolio_load_dataset(
    data_mode="Test",
    assets=["SP500"],
    window_length=150,
    stride=1,
    train_ratio=0.8,
    log_returns=True,
    normalize_mode="zscore"
)

N = min(1000, len(real_set))
real = np.stack([real_set[i][0] for i in range(N)])   # (N, C, 1, T)
real = np.transpose(real.squeeze(2), (0, 2, 1))       # (N, T, C)

# 4) Datos sinteticos
z = torch.randn(N, 100)
with torch.no_grad():
    fake = gen(z).cpu().numpy()                        # (N, C, 1, T)
fake = np.transpose(fake.squeeze(2), (0, 2, 1))        # (N, T, C)

# 5) Visualizacion (PCA o tSNE)
os.makedirs("images", exist_ok=True)
visualization(real, fake, analysis="pca", save_name="sp500_pca")
visualization(real, fake, analysis="tsne", save_name="sp500_tsne")