import numpy as np
import torch
from pydybm.time_series.rnn_gaussian_dybm import RNNGaussianDyBM
from pydybm.base.sgd import RMSProp

class IPMModule:
    """
    Wrapper del RNNGaussianDyBM (= NDyBM de Yu et al.) para el módulo IPM.
    Interfaz limpia entre pydybm (NumPy) y el pipeline PyTorch.
    """
    def __init__(self, m: int, delay: int = 10, n_hidden: int = None):
        # n_hidden = m * 3 según Yu et al. (N = m × 3)
        self.model = RNNGaussianDyBM(
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
        self.m = m

    def reset(self):
        """Llamar al inicio de cada episodio."""
        self.model.init_state()

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        x_t: retornos del paso actual (torch.Tensor, shape: m)
        Devuelve predicción x̂[t+1] como torch.Tensor para concatenar al estado.
        """
        # Convertir a NumPy, formato (1, m) que espera pydybm
        x_np = x_t.detach().cpu().numpy().reshape(1, -1)

        # Obtener predicción ANTES de actualizar (predice t+1 con info hasta t)
        x_hat_np = self.model.predict_next()          # (1, m)

        # Actualizar el modelo online con x[t]
        self.model.learn(x_np)

        # Devolver predicción como tensor (sin gradiente, es input al agente)
        return torch.tensor(x_hat_np.flatten(), dtype=torch.float32)
