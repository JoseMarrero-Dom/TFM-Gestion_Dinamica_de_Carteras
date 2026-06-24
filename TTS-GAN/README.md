# TTS-GAN — Generador de Series Temporales Financieras con GAN Transformer

Este proyecto entrena una Red Generativa Adversarial (GAN) basada en Transformers para generar series temporales sintéticas de activos financieros (SP500, Gold, etc.). El objetivo del TFM es producir datos sintéticos realistas que puedan usarse para aumentar datasets financieros.

---

## Estructura de ficheros

```
TTS-GAN/
├── requirements.txt                  # Dependencias Python del proyecto
└── tts-gan-tfm/
    ├── Train_Portfolio.py            # Punto de entrada principal — GAN conjunta de los 6 activos
    ├── Train_SP500.py                # Lanzador por activo individual (actualmente UST10Y)
    ├── train_GAN.py                  # Lógica completa del bucle de entrenamiento
    ├── GANModels.py                  # Arquitecturas del Generador y Discriminador
    ├── dataLoader.py                 # Carga y preprocesa datos financieros (yfinance + caché)
    ├── cfg.py                        # Todos los argumentos y configuración del experimento
    ├── functions.py                  # Funciones de entrenamiento por epoch (train, train_d)
    ├── adamw.py                      # Implementación manual del optimizador AdamW
    ├── eval.py                       # Evaluación: genera datos sintéticos y visualiza (PCA/t-SNE)
    ├── discriminative_score.py       # Discriminative score (clasificador real vs. sintético)
    ├── grid.py                       # Búsqueda en rejilla de hiperparámetros
    ├── visualizationMetrics.py       # PCA y t-SNE para comparar datos reales vs sintéticos
    ├── data_cache/                   # CSVs cacheados descargados de Yahoo Finance
    │   ├── portfolio_prices_*.csv    # Precios de cierre ajustados (6 activos)
    │   ├── portfolio_open_*.csv      # Apertura (canal intradía)
    │   ├── portfolio_high_*.csv      # Máximo (canal intradía)
    │   ├── portfolio_low_*.csv       # Mínimo (canal intradía)
    │   └── portfolio_vix_*.csv       # Nivel diario del VIX
    ├── images/                       # Dashboards PCA / t-SNE generados por eval.py
    └── utils/
        ├── __init__.py
        ├── utils.py                  # Logging, guardado de checkpoints, estadísticas
        ├── inception_score.py        # Métrica Inception Score (no usada actualmente)
        ├── inception_model.py        # Modelo Inception para IS (no usado actualmente)
        ├── inception.py              # Arquitectura Inception auxiliar
        ├── fid_score.py              # FID score basado en imágenes (no usado actualmente)
        ├── torch_fid_score.py        # FID score adaptado a PyTorch
        └── cal_fid_stat.py           # Calcula estadísticas FID sobre datos reales
```

---

## Qué hace cada fichero

### `Train_Portfolio.py` — Punto de entrada principal

Lanzador de la **GAN conjunta de los 6 activos** (18 canales = 6 activos × 3 canales intradía).
No contiene lógica propia: construye el comando con todos los hiperparámetros y llama a
`train_GAN.py` mediante `os.system(...)`. Es parametrizable por línea de comandos:

```bash
python Train_Portfolio.py --max_iter 100000 --exp_name portfolio
python Train_Portfolio.py --assets SP500 Gold --exp_name sp500_gold   # subconjunto de activos
```

| Argumento | Default | Descripción |
|---|---|---|
| `--assets` | los 6 activos | Lista de activos a entrenar conjuntamente |
| `--max_iter` | 100000 | Iteraciones de entrenamiento |
| `--exp_name` | portfolio | Nombre del experimento (carpeta de logs) |
| `--rank` | 0 | Rank para entrenamiento distribuido |

Hiperparámetros fijados internamente: batch 64, `latent_dim 256`, `g_lr 0.0001`,
`d_lr 0.0002`, loss LSGAN, optimizador Adam (β1=0.9, β2=0.999), `patch_size 15`,
`d_depth 3`, `g_depth 5,4,2`, `--use_intraday`, `--filter_regime moderate stress`,
EMA 0.9999, diff-aug `translation,cutout,color`.

### `Train_SP500.py` — Lanzador por activo individual

Variante que entrena un único activo (a pesar del nombre, **actualmente está configurado
para `UST10Y`**). Misma configuración de hiperparámetros que `Train_Portfolio.py` pero con
`--assets UST10Y` y `--exp_name UST10Y`. Útil como plantilla para entrenar GANs por activo;
basta con editar el activo en el script.

Para pruebas rápidas, reducir `--max_iter` (p. ej. a 500).

---

### `train_GAN.py` — Bucle de entrenamiento

Orquesta todo el proceso de entrenamiento. Flujo principal:

1. Lee los argumentos de `cfg.py`
2. Detecta si hay GPU disponible; si no, usa CPU
3. Instancia `Generator` y `Discriminator` desde `GANModels.py`
4. Carga el dataset con `dataLoader.py`
5. En cada época llama a `train()` de `functions.py`
6. Cada época guarda un checkpoint en `logs/<exp_name>_<timestamp>/Model/checkpoint`
7. Cada época genera un plot de 10 señales sintéticas y lo escribe en TensorBoard

El directorio de salida se crea automáticamente en `logs/` con nombre `<exp_name>_<fecha_hora>`.

---

### `GANModels.py` — Arquitecturas de la GAN

Contiene el **Generador** y el **Discriminador**, ambos basados en Transformers.

#### Generator

Toma un vector de ruido aleatorio `z` de dimensión `latent_dim` (100 por defecto) y produce una señal temporal sintética de forma `(batch, channels, 1, seq_len)`.

Capas:
- `Linear(latent_dim → seq_len × embed_dim)`: expande el ruido al espacio de embeddings
- `pos_embed`: añade codificación posicional aprendida
- `Gen_TransformerEncoder`: pila de `depth` bloques transformer, cada uno con:
  - `LayerNorm` → `MultiHeadAttention` → `Dropout` (conexión residual)
  - `LayerNorm` → `FeedForward` (expansión ×4) → `Dropout` (conexión residual)
- `Conv2d(embed_dim → channels, kernel 1×1)`: proyecta al número de activos/canales

#### Discriminator

Toma una señal temporal real o falsa y produce un escalar (real/falso).

Capas:
- `PatchEmbedding_Linear`: divide la señal en parches de tamaño `patch_size` y los proyecta a `emb_size=50`; añade token CLS y embeddings de posición
- `Dis_TransformerEncoder`: pila de `depth=3` bloques transformer
- `ClassificationHead`: promedia los tokens (`Reduce mean`) → `LayerNorm` → `Linear(50→1)`

#### Bloques compartidos

- `MultiHeadAttention`: atención multi-cabeza estándar con proyecciones Q, K, V
- `ResidualAdd`: wrapper que suma la entrada original a la salida (skip connection)
- `FeedForwardBlock`: MLP de dos capas con GELU y Dropout

---

### `dataLoader.py` — Carga de datos financieros

Clase `portfolio_load_dataset` que extiende `torch.utils.data.Dataset`.

Proceso que sigue al instanciarse:

1. **Descarga precios** de Yahoo Finance para los tickers definidos en `DEFAULT_TICKERS`:
   - SP500 (`^GSPC`), MSCI EAFE (`EFA`), MSCI EM (`EEM`), Gold (`GLD`), Oil WTI (`USO`), US Treasury 10Y (`IEF`)
   - También descarga el VIX (`^VIX`) si se necesita para etiquetas de régimen
2. **Cachea los CSVs** en `data_cache/` para no re-descargar en ejecuciones siguientes
3. **Calcula retornos**: log-retornos (`log(P_t / P_{t-1})`) o porcentaje de cambio
4. **Divide train/test**: por fecha (`split_date`) o por proporción (`train_ratio=0.8`)
5. **Construye ventanas deslizantes** de longitud `window_length` con paso `stride`
6. **Clasifica régimen VIX** (opcional):
   - `low`: VIX < 20 (mercado tranquilo)
   - `moderate`: 20 ≤ VIX < 30
   - `stress`: VIX ≥ 30 (crisis)
7. **Normaliza** cada ventana: z-score o min-max
8. **Formatea** a shape `(N, channels, 1, seq_len)` para ser compatible con la Conv2d del generador

Parámetros clave:
| Parámetro | Default | Descripción |
|---|---|---|
| `assets` | todos | Lista de activos a incluir (ej. `["SP500"]`) |
| `window_length` | 150 | Días por ventana |
| `stride` | 1 | Paso entre ventanas consecutivas |
| `log_returns` | True | Usar log-retornos |
| `normalize_mode` | zscore | Normalización por ventana |
| `filter_regime` | None | Filtrar solo ventanas de régimen específico |

---

### `cfg.py` — Configuración y argumentos

Define todos los hiperparámetros del experimento mediante `argparse`. Los más relevantes:

| Argumento | Valor en los lanzadores | Descripción |
|---|---|---|
| `--max_iter` | 100 000 | Número total de iteraciones de entrenamiento |
| `--batch_size` | 64 | Muestras por batch |
| `--latent_dim` | 256 | Dimensión del vector de ruido z |
| `--g_lr` | 0.0001 | Learning rate del generador |
| `--d_lr` | 0.0002 | Learning rate del discriminador |
| `--loss` | lsgan | Función de pérdida (lsgan / hinge / wgangp) |
| `--patch_size` | 15 | Tamaño del parche en el discriminador (divide seq_len=150) |
| `--df_dim` | 384 | Dimensión base del discriminador |
| `--d_depth` | 3 | Número de bloques transformer en el discriminador |
| `--g_depth` | 5,4,2 | Profundidad del generador |
| `--dropout` | 0 | Dropout (0 = desactivado) |
| `--ema` | 0.9999 | Exponential Moving Average del generador |
| `--exp_name` | portfolio / UST10Y | Nombre del experimento (carpeta de logs) |
| `--assets` | los 6 activos | Activos financieros a entrenar (conjuntamente) |
| `--use_intraday` | activado | Usa 3 canales por activo (log-ret, open-close, high-low) |
| `--filter_regime` | moderate stress | Entrena solo con ventanas de esos regímenes de VIX |
| `--normalize_mode` | zscore | Normalización por ventana |

---

### `functions.py` — Funciones de entrenamiento

Contiene el código que se ejecuta en cada iteración dentro del bucle de `train_GAN.py`.

#### `train(args, gen_net, dis_net, ...)`

Entrena tanto el generador como el discriminador en cada iteración:

1. Mueve el batch real al dispositivo correcto (GPU/CPU)
2. Genera ruido `z` aleatorio → produce imágenes falsas con el generador
3. **Entrena el discriminador**: calcula `d_loss` comparando real vs falso
4. **Entrena el generador**: genera nuevo batch falso, calcula `g_loss` (engañar al discriminador)
5. Actualiza EMA (Exponential Moving Average) de los pesos del generador

#### `train_d(args, gen_net, dis_net, ...)`

Versión que solo entrena el discriminador (sin actualizar el generador). Comentada en el flujo actual.

#### `compute_gradient_penalty(D, real, fake, phi)`

Calcula la penalización de gradiente para la variante WGAN-GP (no activa con LSGAN).

#### `LinearLrDecay`

Clase que reduce linealmente el learning rate desde `start_lr` hasta 0 a lo largo del entrenamiento.

#### `load_params` / `copy_params`

Gestionan los pesos del generador EMA: copian parámetros entre el modelo activo y el promedio móvil para evaluar con los pesos suavizados.

---

### `adamw.py` — Optimizador AdamW

Implementación manual del optimizador AdamW (Adam con decaimiento de pesos desacoplado). Se usa como alternativa a `torch.optim.Adam` cuando se pasa `--optimizer adamw`. La diferencia con Adam estándar es que el weight decay se aplica directamente a los pesos antes del paso del gradiente, no a través del gradiente.

---

### `eval.py` — Evaluación post-entrenamiento

Script independiente que se ejecuta después del entrenamiento para visualizar resultados.

Pasos:
1. **Detecta automáticamente** el checkpoint más reciente dentro de `logs/` (no hay que
   editar rutas manualmente; imprime por consola el checkpoint que usa).
2. Infiere la configuración del generador (canales, latent_dim, seq_len) desde el checkpoint.
3. Carga datos reales de test (por defecto N=1000 ventanas, regímenes moderate/stress).
4. Genera muestras sintéticas con ruido aleatorio.
5. Llama a `visualizationMetrics.py` para generar dashboards PCA y t-SNE.

Salidas en `images/` (formato `.png`), por ejemplo `<exp>_<timestamp>_dashboard.png` y
`<exp>_<timestamp>_pca_tsne.png` (y por activo si la GAN es conjunta).

### `discriminative_score.py` — Discriminative score

Entrena un clasificador para distinguir muestras reales de sintéticas. Un score cercano a
**0.5** indica que el clasificador no las diferencia (alta calidad del generador); cercano a
**1.0** indica que las muestras sintéticas son fácilmente detectables (baja calidad).

### `grid.py` — Búsqueda de hiperparámetros

Lanza múltiples entrenamientos variando hiperparámetros para comparar configuraciones del GAN.

---

### `visualizationMetrics.py` — Métricas visuales

Función `visualization(ori_data, generated_data, analysis, save_name)` que compara distribuciones de datos reales (rojo) y sintéticos (azul) proyectadas en 2D.

- `analysis='pca'`: Análisis de Componentes Principales
- `analysis='tsne'`: t-SNE (más lento, más informativo)

Guarda el gráfico como PNG en `./images/<save_name>.png`.

---

### `utils/utils.py` — Utilidades generales

- `set_log_dir(root, exp_name)`: crea la estructura de carpetas del experimento (`Model/`, `Log/`, `Samples/`)
- `save_checkpoint(states, is_best, output_dir)`: guarda el estado del modelo en disco
- `create_logger(log_dir)`: configura logging a fichero y consola
- `make_grid` / `save_image`: funciones de visualización de grids de imágenes (copiadas de torchvision para compatibilidad)
- `RunningStats`: calcula media y varianza en una ventana deslizante de forma eficiente

### `utils/torch_fid_score.py` y demás utils

Código para calcular FID (Fréchet Inception Distance) y Inception Score. Son métricas estándar para evaluar GANs de imágenes. Actualmente están comentadas en el flujo principal porque estas métricas no aplican directamente a series temporales; están presentes como infraestructura para posibles adaptaciones futuras.

---

## Instalación

```bash
cd TTS-GAN
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

El requirements incluye PyTorch con soporte CUDA. Si solo se tiene CPU o AMD:
```bash
# CPU puro
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# AMD ROCm (no funciona en WSL2, solo Linux nativo)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
```

---

## Entrenamiento

```bash
cd tts-gan-tfm

# GAN conjunta de los 6 activos (entrada principal)
python Train_Portfolio.py --max_iter 100000 --exp_name portfolio

# (alternativa) GAN de un activo individual
python Train_SP500.py
```

El script detecta automáticamente si hay GPU disponible; si no, usa CPU.

Los resultados se guardan en:
```
tts-gan-tfm/logs/<exp_name>_<YYYY_MM_DD_HH_MM_SS>/
    Model/checkpoint          # pesos del modelo (se sobreescribe cada época)
    Log/                      # ficheros de log y TensorBoard
    Samples/                  # imágenes de muestra (si --show activo)
```

Para monitorizar el entrenamiento en TensorBoard:
```bash
tensorboard --logdir tts-gan-tfm/logs/
```

Para una prueba rápida (pocas iteraciones), reducir `--max_iter` (p. ej. a 500).

---

## Evaluación

Una vez terminado el entrenamiento, ejecutar directamente:

```bash
python eval.py                  # PCA / t-SNE + dashboards
python discriminative_score.py  # discriminative score (real vs. sintético)
```

`eval.py` detecta automáticamente el experimento más reciente dentro de `logs/` y carga su
checkpoint, sin necesidad de editar ninguna ruta. Imprime por consola el checkpoint que usa.

Genera en `images/` los dashboards `.png` (`<exp>_<timestamp>_dashboard.png` y
`<exp>_<timestamp>_pca_tsne.png`) comparando la distribución de series reales vs. sintéticas.

---

## Cómo funciona la GAN (resumen)

Una GAN tiene dos redes que compiten entre sí:

- El **Generador** recibe ruido aleatorio y aprende a producir series temporales que parezcan reales.
- El **Discriminador** recibe series (reales o generadas) y aprende a distinguirlas.

En cada iteración:
1. El discriminador se entrena con series reales (etiqueta 1) y series falsas del generador (etiqueta 0).
2. El generador se entrena para que sus series sean clasificadas como reales por el discriminador.

Con LSGAN, en lugar de usar entropía cruzada, se minimiza el error cuadrático medio, lo que da entrenamiento más estable. El generador EMA (media móvil de pesos) produce resultados más suaves y estables que el generador instantáneo.

---

## Datos

Los datos se descargan automáticamente de Yahoo Finance al primer uso y se cachean en `data_cache/`. El rango por defecto es 2004-01-01 a 2025-12-31 (el histórico común efectivo arranca en abril de 2006 por la fecha de inicio de algunos ETFs). Con `--use_intraday` se trabaja con **3 canales por activo** (log-retorno close-to-close, rango open-close y rango high-low), normalizados por ventana.

Los ficheros CSV ya presentes en `data_cache/` contienen:
- `portfolio_prices_*.csv`: precios de cierre ajustados de los 6 activos
- `portfolio_open_*.csv` / `portfolio_high_*.csv` / `portfolio_low_*.csv`: OHLC para los canales intradía
- `portfolio_vix_*.csv`: nivel diario del índice VIX (volatilidad implícita del mercado)
