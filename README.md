# TFM — Gestión de carteras con aumento de datos sintéticos (TTS-GAN) y aprendizaje por refuerzo (PPO)

Pipeline de investigación para la optimización de una cartera multi-activo que combina
tres fases: **análisis exploratorio de datos (EDA) → generación de datos sintéticos con
TTS-GAN → optimización de la asignación con PPO**, validado mediante un esquema
**walk-forward** sobre el período 2021–2025.

> **Reproducibilidad:** los datos utilizados se descargan automáticamente vía Yahoo Finance
> (con caché local en CSV ya incluida en el repositorio) y **se pueden reproducir los
> resultados ejecutando los scripts de Python siguiendo la guía en README.md del GitHub**.

---

## Universo de activos

La cartera está compuesta por 6 activos (proxies vía ETF) más efectivo (*cash*),
descargados de Yahoo Finance para el período **2004-01-01 → 2025-12-31** (el histórico
común efectivo arranca en abril de 2006 por la fecha de inicio de algunos ETFs):

| Activo | Ticker | Descripción |
|---|---|---|
| `SP500` | `^GSPC` | S&P 500 |
| `MSCI_EAFE` | `EFA` | iShares MSCI EAFE ETF |
| `MSCI_EM` | `EEM` | iShares MSCI Emerging Markets ETF |
| `Gold` | `GLD` | SPDR Gold Shares |
| `Oil_WTI` | `USO` | United States Oil Fund (proxy WTI) |
| `UST10Y` | `IEF` | iShares 7-10 Year Treasury Bond ETF |
| Régimen | `^VIX` | Índice de volatilidad (clasificación de régimen) |

---

## Estructura del repositorio

```
TFM/
├── TFM_EDA/                      # FASE 1 — Análisis exploratorio de datos
│   ├── EDA_Portfolio_TFM.ipynb   #   Notebook principal del EDA
│   ├── regenerar_figuras_leyendas.py  #   Regenera fig6/fig7/fig9 con leyendas claras
│   ├── figuras_eda/              #   Figuras y tablas generadas (.png, .csv)
│   └── requirements.txt
│
├── TTS-GAN/                      # FASE 2 — GAN de series temporales (Transformer)
│   ├── tts-gan-tfm/
│   │   ├── GANModels.py          #   Generator / Discriminator (Transformer)
│   │   ├── train_GAN.py          #   Bucle de entrenamiento de bajo nivel
│   │   ├── Train_Portfolio.py    #   Lanzador: GAN conjunta de los 6 activos
│   │   ├── Train_SP500.py        #   Lanzador: GAN por activo (ejemplo)
│   │   ├── dataLoader.py         #   Descarga yfinance + caché + ventanas
│   │   ├── eval.py / discriminative_score.py / visualizationMetrics.py
│   │   ├── data_cache/           #   CSVs cacheados (precios/OHLC/VIX)
│   │   ├── images/               #   Dashboards PCA / t-SNE por activo
│   │   └── logs/                 #   Checkpoints de entrenamiento (generados)
│   └── requirements.txt
│
├── RL/                           # FASE 3 — Aprendizaje por refuerzo (PPO)
│   ├── Environment/
│   │   ├── environment.py        #   Entorno Gymnasium base
│   │   └── environment_IPM.py    #   Entorno con módulo de predicción IPM
│   ├── PPO/agent.py              #   Wrapper de PPO (Stable-Baselines3)
│   ├── IPM/
│   │   ├── ipm.py                #   IPMModule (N-DyBM: RNNGaussianDyBM de pydybm)
│   │   ├── train_ndybm.py        #   Entrenamiento del N-DyBM
│   │   └── dybm/                 #   Librería pydybm (instalada como editable)
│   ├── train_PPO_agent.py        #   Entrena PPO + IPM con datos reales
│   ├── train_PPO_gan_augmented.py#   Entrena PPO baseline vs. aumentado con GAN
│   ├── results_gan_augmented/    #   Resultados (modelos, métricas, gráficas)
│   ├── results_gan_10_000_000/   #   Resultados con 10M steps
│   ├── data_cache/               #   CSVs cacheados
│   └── requirements.txt
│
├── Evaluation/                   # Validación temporal (walk-forward)
│   ├── walk-forward-2021.py      #   Evaluación 2021–2025 por año / régimen / evento
│   └── data_cache/               #   CSVs cacheados
│
├── crear_proyecto.sh             # Script de scaffolding del proyecto
└── README.md
```

---

## Requisitos

- **Python 3.12**
- **Git**
- (Opcional) GPU NVIDIA con CUDA para acelerar el entrenamiento de TTS-GAN y PPO.
  Todo funciona también en CPU (más lento).

Cada módulo (`TFM_EDA`, `TTS-GAN`, `RL`) tiene su propio `requirements.txt` y se recomienda
**un entorno virtual independiente por módulo**, ya que las versiones de PyTorch difieren entre
TTS-GAN y RL.

### Datos

No es necesario descargar nada manualmente. El cargador
([dataLoader.py](TTS-GAN/tts-gan-tfm/dataLoader.py)) usa `yfinance` y guarda los CSVs en
`data_cache/`. Si el CSV ya existe (como ocurre en este repositorio), **se lee de la caché**;
solo se vuelve a descargar de Yahoo Finance si borras la caché o cambias el rango de fechas.

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/JoseMarrero-Dom/tfm.git
cd tfm
```

Crea un entorno virtual por módulo. Por ejemplo, para el EDA:

```bash
cd TFM_EDA
python3.12 -m venv venv
source venv/bin/activate          # Linux / Mac
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
deactivate
cd ..
```

Repite el mismo procedimiento para `TTS-GAN/` y `RL/`:

```bash
cd TTS-GAN && python3.12 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && deactivate && cd ..
cd RL      && python3.12 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && deactivate && cd ..
```

> El entorno de `RL` instala automáticamente la librería **pydybm** (N-DyBM) como dependencia
> editable desde `RL/IPM/dybm` (declarada en `RL/requirements.txt`).
>
> El módulo `Evaluation/` reutiliza el entorno de `RL` (mismas dependencias: gymnasium,
> stable-baselines3, torch, pandas, matplotlib).

---

## Reproducción de los resultados (paso a paso)

Ejecuta las fases en orden. Cada script debe lanzarse **desde el directorio de su módulo**
y con el entorno virtual correspondiente activado.

### Fase 1 — Análisis exploratorio de datos (EDA)

```bash
cd TFM_EDA
source venv/bin/activate
jupyter lab            # abre y ejecuta EDA_Portfolio_TFM.ipynb
```

El notebook descarga/lee los datos, calcula estadísticos (sesgo, curtosis, Jarque-Bera,
ADF, Ljung-Box) y genera todas las figuras y tablas en `figuras_eda/`.

Para regenerar únicamente las figuras de distribuciones, Q-Q y ACF (con leyendas legibles):

```bash
python regenerar_figuras_leyendas.py
```

### Fase 2 — Entrenamiento del TTS-GAN

```bash
cd TTS-GAN/tts-gan-tfm
source ../venv/bin/activate

# GAN conjunta de los 6 activos (18 canales intraday)
python Train_Portfolio.py --max_iter 100000 --exp_name portfolio

# (alternativa) GAN por activo individual
python Train_SP500.py
```

Los `Train_*.py` componen y lanzan internamente `train_GAN.py` con la configuración del
modelo (Transformer, LSGAN, EMA, diff-aug, etc.). Los **checkpoints** se guardan en
`TTS-GAN/tts-gan-tfm/logs/<exp_name>/` y son la entrada de la fase de aumento de datos.

Evaluación de la calidad de las muestras sintéticas (PCA / t-SNE / discriminative score):

```bash
python eval.py
python discriminative_score.py
```

### Fase 3 — Entrenamiento del agente PPO

```bash
cd RL
source venv/bin/activate
```

**Opción A — PPO + IPM con datos reales** (baseline con módulo N-DyBM):

```bash
python train_PPO_agent.py
# Genera: ppo_portfolio.zip, eval_metrics.png, eval_metrics_weights.png
```

**Opción B — PPO baseline vs. aumentado con datos sintéticos del GAN:**

```bash
python train_PPO_gan_augmented.py
# Lee los checkpoints de TTS-GAN/tts-gan-tfm/logs/
# Genera en results_gan_augmented/:
#   ppo_baseline.zip, ppo_augmented.zip,
#   vec_normalize_*.pkl, eval_*.png, comparison.csv
```

El entrenamiento usa PPO (`MlpPolicy`) de Stable-Baselines3 con `VecNormalize`. La recompensa
se basa en el ratio de Sharpe con penalización por rotación de cartera.

### Fase 4 — Validación walk-forward (2021–2025)

```bash
cd Evaluation
source ../RL/venv/bin/activate

# Listar los modelos disponibles en una carpeta de resultados
python walk-forward-2021.py --model_dir ../RL/results_gan_augmented --list

# Evaluar un modelo concreto (o el más reciente si se omite --model_name)
python walk-forward-2021.py \
    --model_dir ../RL/results_gan_augmented \
    --model_name ppo_augmented
```

Genera, en la carpeta del modelo, las métricas y gráficas walk-forward:
`wf_<modelo>_metrics.csv`, `wf_<modelo>_equity.png`, `wf_<modelo>_tabla.png` y los
boxplots de la cartera por año. Las métricas se desglosan **globalmente, por año, por
régimen de VIX (low / moderate / stress) y por evento de mercado**.

---

## Uso

Con cualquiera de los entornos activados puedes lanzar JupyterLab para inspección
interactiva:

```bash
jupyter lab
```

---

## Notas sobre reproducibilidad

- Las semillas están fijadas (`SEED = 42` / `seed=0`) en los scripts de entrenamiento y
  generación para favorecer la reproducibilidad.
- Los resultados publicados en `RL/results_gan_augmented/` y `RL/results_gan_10_000_000/`
  se generaron con los datos cacheados incluidos en el repositorio.
- El entrenamiento del TTS-GAN y de PPO con muchos pasos (p. ej. 10M) es intensivo; se
  recomienda GPU. Para una ejecución rápida de prueba, reduce `--max_iter` (GAN) y
  `total_timesteps` (PPO).
```
