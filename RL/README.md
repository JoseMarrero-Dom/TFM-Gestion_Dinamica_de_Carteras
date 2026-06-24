# RL — Aprendizaje por Refuerzo para Gestión de Cartera (Fase 3)

Entrena un agente **PPO** (Stable-Baselines3) sobre un entorno **Gymnasium** que modela la
asignación dinámica de una cartera de 6 activos + efectivo. Incluye un módulo de predicción
**IPM (N-DyBM)** y el pipeline de **aumento de datos** con muestras sintéticas del TTS-GAN.

Forma parte del proyecto global descrito en el [README raíz](../README.md).

---

## Contenido

```
RL/
├── Environment/
│   ├── environment.py            # Entorno Gymnasium base (PortfolioEnv)
│   └── environment_IPM.py        # Entorno con predicción IPM en la observación
├── PPO/
│   └── agent.py                  # PPOAgent: wrapper de PPO (MlpPolicy) de SB3
├── IPM/
│   ├── ipm.py                    # IPMModule — wrapper de RNNGaussianDyBM (N-DyBM)
│   ├── train_ndybm.py            # Entrenamiento del N-DyBM
│   └── dybm/                     # Librería pydybm (instalada como dependencia editable)
├── train_PPO_agent.py            # Entrena PPO + IPM con datos reales
├── train_PPO_gan_augmented.py    # Entrena PPO baseline vs. aumentado con GAN
├── results_gan_augmented/        # Resultados (modelos, vec_normalize, métricas, gráficas)
├── results_gan_10_000_000/       # Resultados con 10M timesteps
├── data_cache/                   # CSVs cacheados (precios / OHLC / VIX)
├── ppo_portfolio.zip             # Modelo entrenado (Opción A)
└── requirements.txt
```

---

## El entorno (Gymnasium)

- `action_space`: `Box(0, 1, shape=(7,))` — asignación entre 6 activos + cash (se normaliza a suma 1).
- `observation_space`: `Box(-inf, inf, shape=(obs_dim,))` con log-returns en ventana, rangos
  intradía (open-close, high-low), VIX, predicción IPM y pesos actuales.
- **Recompensa**: ratio de Sharpe con penalización por rotación (coste de transacción).
- Cada paso representa un rebalanceo; los episodios se definen por semanas (`episode_weeks`).

## El módulo IPM / N-DyBM

`IPMModule` ([IPM/ipm.py](IPM/ipm.py)) envuelve el `RNNGaussianDyBM` de **pydybm** (= NDyBM de
Yu et al.) y aporta una predicción del siguiente estado de mercado que se concatena al vector
de observación. La librería `pydybm` se instala como dependencia editable desde `IPM/dybm`.

---

## Requisitos

- Python 3.12 (GPU NVIDIA/CUDA opcional y recomendada)
- Dependencias en `requirements.txt`: `gymnasium`, `stable_baselines3`, `torch`, `pandas`,
  `matplotlib`, `scikit-learn` y `pydybm` (editable desde `IPM/dybm`).

---

## Instalación

```bash
cd RL
python3.12 -m venv venv
source venv/bin/activate          # Linux / Mac
# venv\Scripts\activate           # Windows
pip install -r requirements.txt   # instala también pydybm (N-DyBM) en modo editable
```

---

## Uso

> Ejecuta los scripts **desde este directorio** (`RL/`): usan rutas relativas a `./data_cache`
> y a `../TTS-GAN/tts-gan-tfm`.

**Opción A — PPO + IPM con datos reales:**

```bash
python train_PPO_agent.py
# Genera: ppo_portfolio.zip, eval_metrics.png, eval_metrics_weights.png
```

**Opción B — PPO baseline vs. aumentado con datos sintéticos del GAN:**

```bash
python train_PPO_gan_augmented.py
# Requiere checkpoints en ../TTS-GAN/tts-gan-tfm/logs/ (ver README de TTS-GAN)
# Genera en results_gan_augmented/:
#   ppo_baseline.zip, ppo_augmented.zip,
#   vec_normalize_baseline.pkl, vec_normalize_augmented.pkl,
#   eval_baseline.png, eval_augmented.png, comparison.csv
```

**Entrenar solo el N-DyBM (IPM):**

```bash
python IPM/train_ndybm.py
```

La validación temporal (walk-forward) de los modelos entrenados aquí se realiza desde el
módulo [`Evaluation/`](../Evaluation/README.md).

---

## Datos

Los CSVs de `data_cache/` (precios, open/high/low, VIX; 2006–2025) ya están incluidos. El
cargador subyacente ([../TTS-GAN/tts-gan-tfm/dataLoader.py](../TTS-GAN/tts-gan-tfm/dataLoader.py))
descarga de Yahoo Finance solo si la caché no existe.
