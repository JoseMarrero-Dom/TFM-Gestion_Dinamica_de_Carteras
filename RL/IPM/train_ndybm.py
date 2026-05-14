# import dataLoader from TTS-GAN/tts-gan-tfm/dataLoader.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pydybm.time_series.rnn_gaussian_dybm import RNNGaussianDyBM
from pydybm.base.sgd import RMSProp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../TTS-GAN/tts-gan-tfm"))
sys.path.insert(0, ROOT)
from dataLoader import portfolio_load_dataset

dataset = portfolio_load_dataset(
    data_mode="Train",
    is_normalize=False,
    log_returns=True,
    use_intraday=True,
    use_windows=False  # queremos la serie completa para el DyBM
)

# Imprimir primeras filas del dataset para verificar
print("Shape del dataset:", dataset.data.shape)
print("Primeras filas del dataset:")
# Mostrar headers y primeras filas
print(dataset.data[:1])  # muestra las primeras 5 filas del dataset

m = dataset.data.shape[1]  # canales = 3 * n_activos
model = RNNGaussianDyBM(
    in_dim=m,
    out_dim=m,
    rnn_dim=m * 3,
    spectral_radius=0.95,
    sparsity=0.1,
    delay=10,
    decay_rates=[0.5, 0.9],
    SGD=RMSProp(rate=1e-3, gamma=0.9),
    leak=1.0
)

losses = []
for window, _ in dataset:
    # window: (m, 1, seq_len)
    seq = window[:, 0, :].T  # (seq_len, m)
    model.init_state()
    for t in range(seq.shape[0]):
        x_t = seq[t].reshape(1, -1)
        pred = model.predict_next()          # predice usando el estado actual
        model.learn(x_t)
        losses.append(np.mean((pred - x_t) ** 2))


def eval_sequence(model, window, warmup=10):
    # window: (m, 1, T)
    seq = window[:, 0, :].T  # (T, m)
    preds = []
    trues = []
    model.init_state()
    for t in range(seq.shape[0]):
        pred = model.predict_next().reshape(-1)  # (m,)
        x_t = seq[t]
        model.learn(x_t.reshape(1, -1))
        if t >= warmup:
            preds.append(pred)
            trues.append(x_t)
    preds = np.stack(preds)
    trues = np.stack(trues)

    mse = ((preds - trues) ** 2).mean(axis=0)
    rmse = np.sqrt(mse)
    mae = np.abs(preds - trues).mean(axis=0)
    return preds, trues, mse, rmse, mae


# Evaluacion en TRAIN
train_window, _ = dataset[0]  # use_windows=False -> solo una ventana
preds, trues, mse, rmse, mae = eval_sequence(model, train_window, warmup=10)
print("RMSE promedio:", rmse.mean())

# Visualizar 3 canales del primer activo (indices 0,1,2)
plt.figure(figsize=(10, 6))
for i, idx in enumerate([0, 1, 2]):
    plt.subplot(3, 1, i + 1)
    plt.plot(trues[-1000:, idx], label="real")
    plt.plot(preds[-1000:, idx], label="pred")
    plt.legend()
plt.tight_layout()
plt.show()