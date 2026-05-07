# TTS-GAN — Generador de Series Temporales Financieras con GAN Transformer

Este proyecto entrena una Red Generativa Adversarial (GAN) basada en Transformers para generar series temporales sintéticas de activos financieros (SP500, Gold, etc.). El objetivo del TFM es producir datos sintéticos realistas que puedan usarse para aumentar datasets financieros.

---

## Estructura de ficheros

```
TTS-GAN/
├── requirements.txt                  # Dependencias Python del proyecto
└── tts-gan-tfm/
    ├── Train_SP500.py                # Punto de entrada principal — lanza el entrenamiento
    ├── train_GAN.py                  # Lógica completa del bucle de entrenamiento
    ├── GANModels.py                  # Arquitecturas del Generador y Discriminador
    ├── dataLoader.py                 # Carga y preprocesa datos financieros (SP500, etc.)
    ├── dataLoaderUniMB.py            # DataLoader alternativo para dataset UniMiB (acelerómetros)
    ├── cfg.py                        # Todos los argumentos y configuración del experimento
    ├── functions.py                  # Funciones de entrenamiento por epoch (train, train_d)
    ├── adamw.py                      # Implementación manual del optimizador AdamW
    ├── eval.py                       # Script de evaluación: genera datos sintéticos y visualiza
    ├── visualizationMetrics.py       # PCA y t-SNE para comparar datos reales vs sintéticos
    ├── data_cache/                   # CSVs cacheados de precios descargados de Yahoo Finance
    │   ├── portfolio_prices_*.csv
    │   └── portfolio_vix_*.csv
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

### `Train_SP500.py` — Punto de entrada

Es el script que el usuario ejecuta. No contiene lógica propia: construye el comando con todos los hiperparámetros y llama a `train_GAN.py` mediante `os.system(...)`.

Parámetros que fija:
- Dataset: `portfolio`, activo: `SP500`
- Batch size: 16 (generador y discriminador)
- Iteraciones máximas: 500 000
- Dimensión latente: 100 (tamaño del vector de ruido de entrada al generador)
- Learning rate generador: 0.0001 / discriminador: 0.0003
- Loss: LSGAN (Least Squares GAN)
- Optimizador: Adam (β1=0.9, β2=0.999)

Para pruebas rápidas, cambiar `--max_iter 500000` a `--max_iter 500`.

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

| Argumento | Default en Train_SP500 | Descripción |
|---|---|---|
| `--max_iter` | 500 000 | Número total de iteraciones de entrenamiento |
| `--batch_size` | 16 | Muestras por batch |
| `--latent_dim` | 100 | Dimensión del vector de ruido z |
| `--g_lr` | 0.0001 | Learning rate del generador |
| `--d_lr` | 0.0003 | Learning rate del discriminador |
| `--loss` | lsgan | Función de pérdida (lsgan / hinge / wgangp) |
| `--patch_size` | 2 | Tamaño del parche en el discriminador |
| `--df_dim` | 384 | Dimensión base del discriminador |
| `--d_depth` | 3 | Número de bloques transformer en el discriminador |
| `--g_depth` | 5,4,2 | Profundidad del generador |
| `--dropout` | 0 | Dropout (0 = desactivado) |
| `--ema` | 0.9999 | Exponential Moving Average del generador |
| `--exp_name` | Running | Nombre del experimento (carpeta de logs) |
| `--window_length` | 150 (default) | Longitud de ventana temporal |
| `--asset` | SP500 | Activo financiero a entrenar |

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
1. Carga un checkpoint guardado en `logs/`
2. Instancia el generador con los mismos parámetros
3. Carga datos reales del conjunto de test
4. Genera N=1000 muestras sintéticas con ruido aleatorio
5. Llama a `visualizationMetrics.py` para generar gráficos PCA y t-SNE

Hay que actualizar la variable `ckpt_path` con la ruta al checkpoint generado en el entrenamiento.

---

### `visualizationMetrics.py` — Métricas visuales

Función `visualization(ori_data, generated_data, analysis, save_name)` que compara distribuciones de datos reales (rojo) y sintéticos (azul) proyectadas en 2D.

- `analysis='pca'`: Análisis de Componentes Principales
- `analysis='tsne'`: t-SNE (más lento, más informativo)

Guarda el gráfico como PDF en `./images/<save_name>.pdf`.

---

### `dataLoaderUniMB.py` — DataLoader alternativo (no activo)

Carga el dataset UniMiB-SHAR de señales de acelerómetro. Era el dataset original del paper base antes de la adaptación financiera. No se usa en el flujo SP500, pero está disponible si se quiere entrenar con datos de actividad humana.

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
python3 -m venv venv
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
python Train_SP500.py
```

El script detecta automáticamente si hay GPU disponible; si no, usa CPU.

Los resultados se guardan en:
```
tts-gan-tfm/logs/Running_<YYYY_MM_DD_HH_MM_SS>/
    Model/checkpoint          # pesos del modelo (se sobreescribe cada época)
    Log/                      # ficheros de log y TensorBoard
    Samples/                  # imágenes de muestra (si --show activo)
```

Para monitorizar el entrenamiento en TensorBoard:
```bash
tensorboard --logdir tts-gan-tfm/logs/
```

Para una prueba rápida (pocas iteraciones):
```bash
# Editar Train_SP500.py y cambiar --max_iter 500000 a --max_iter 500
python Train_SP500.py
```

---

## Evaluación

Una vez terminado el entrenamiento, ejecutar directamente:

```bash
python eval.py
```

El script detecta automáticamente el experimento más reciente dentro de `logs/` y carga su checkpoint, sin necesidad de editar ninguna ruta. Imprime por consola el checkpoint que está usando.

Genera `images/sp500_pca.pdf` y `images/sp500_tsne.pdf` comparando la distribución de series reales vs sintéticas.

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

Los datos se descargan automáticamente de Yahoo Finance al primer uso y se cachean en `data_cache/`. El rango por defecto es 2004-01-01 a 2025-12-31. Se trabaja con log-retornos diarios normalizados por ventana.

Los ficheros CSV ya presentes en `data_cache/` contienen:
- `portfolio_prices_*.csv`: precios de cierre ajustados de los 6 activos
- `portfolio_vix_*.csv`: nivel diario del índice VIX (volatilidad implícita del mercado)
