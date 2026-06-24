# Evaluation — Validación Temporal Walk-Forward

Evalúa los agentes PPO entrenados en [`RL/`](../RL/README.md) sobre datos **fuera de muestra
(2021–2025)** mediante un esquema **walk-forward**, desglosando el rendimiento por año, por
régimen de volatilidad (VIX) y por eventos de mercado.

Forma parte del proyecto global descrito en el [README raíz](../README.md).

---

## Contenido

```
Evaluation/
├── walk-forward-2021.py    # Script de evaluación walk-forward 2021–2025
└── data_cache/             # CSVs cacheados (precios / OHLC / VIX)
```

---

## Qué calcula

A partir de un modelo PPO (`.zip`) y su `vec_normalize_*.pkl`, reconstruye la cartera día a día
sobre el período de test y calcula métricas (retorno, volatilidad, Sharpe, drawdown, etc.):

- **Global** 2021–2025.
- **Por año** (2021, 2022, 2023, 2024, 2025).
- **Por régimen de VIX**: low (<20), moderate (20-30), stress (>30).
- **Por evento de mercado** (episodios explícitos definidos en el script).

Salidas en la carpeta del modelo evaluado: `wf_<modelo>_metrics.csv`,
`wf_<modelo>_equity.png`, `wf_<modelo>_tabla.png` y boxplots de la cartera por año.

---

## Requisitos

Reutiliza el entorno virtual de [`RL/`](../RL/README.md) (gymnasium, stable-baselines3, torch,
pandas, matplotlib). No necesita un `requirements.txt` propio.

```bash
cd Evaluation
source ../RL/venv/bin/activate
```

---

## Uso

```bash
# Listar los modelos disponibles en una carpeta de resultados
python walk-forward-2021.py --model_dir ../RL/results_gan_augmented --list

# Evaluar un modelo concreto (si se omite --model_name, usa el más reciente)
python walk-forward-2021.py \
    --model_dir ../RL/results_gan_augmented \
    --model_name ppo_augmented
```

### Argumentos

| Argumento | Descripción |
|---|---|
| `--model_dir` | (Obligatorio) Carpeta con el `.zip` del modelo y su `vec_normalize_*.pkl`. |
| `--model_name` | Nombre del modelo sin extensión. Por defecto, el más reciente de la carpeta. |
| `--cache_dir` | Carpeta de datos cacheados. Por defecto `../RL/data_cache`. |
| `--list` | Lista los modelos disponibles en `--model_dir` y termina. |

---

## Datos

Los CSVs de `data_cache/` (2006–2025) ya están incluidos; el período de test usado es
2021-01-01 en adelante. Sin necesidad de descarga.
