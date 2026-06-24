# TFM_EDA — Análisis Exploratorio de Datos (Fase 1)

Análisis exploratorio de la cartera multi-activo: caracteriza estadísticamente los retornos
diarios de los 6 activos antes de las fases de generación sintética (TTS-GAN) y aprendizaje
por refuerzo (PPO). Aporta la evidencia empírica (fat tails, asimetría, agrupamiento de
volatilidad, regímenes de mercado) que justifica el diseño del resto del pipeline.

Forma parte del proyecto global descrito en el [README raíz](../README.md).

---

## Contenido

```
TFM_EDA/
├── EDA_Portfolio_TFM.ipynb          # Notebook principal del EDA
├── regenerar_figuras_leyendas.py    # Regenera fig6/fig7/fig9 con leyendas legibles
├── figuras_eda/                     # Figuras (.png) y tablas (.csv) generadas
│   ├── fig1_precios_normalizados.png
│   ├── fig2_retornos_diarios.png
│   ├── fig3_volatilidad_rolling.png
│   ├── fig4_vix_regimen.png
│   ├── fig5_correlaciones.png
│   ├── fig6_distribuciones.png      # Empírica vs. Normal vs. t-Student
│   ├── fig7_qqplots.png             # Q-Q plots (desviación de la normalidad)
│   ├── fig8_drawdown.png
│   ├── fig9_acf_cuadrados.png       # ACF de r² (efecto GARCH)
│   ├── tabla_estadisticos_descriptivos.csv
│   ├── tabla_stats_por_regimen.csv
│   └── correlaciones_*.csv          # Por régimen: global / baja vol. / estrés
└── requirements.txt
```

---

## Qué hace el notebook

- Descarga (o lee de caché) los precios de Yahoo Finance y calcula **retornos logarítmicos**.
- Estadísticos descriptivos anualizados: media, volatilidad, **sesgo, curtosis**, Sharpe.
- Tests de hipótesis: **Jarque-Bera** (normalidad), **ADF** (estacionariedad) y
  **Ljung-Box** sobre r² (efecto ARCH/GARCH).
- Clasificación de **régimen de mercado** según el VIX (baja volatilidad / moderada / estrés)
  y estadísticos por régimen.
- Genera todas las figuras y tablas en `figuras_eda/`.

---

## Requisitos

- Python 3.12
- Dependencias en `requirements.txt` (numpy, pandas, matplotlib, seaborn, scipy,
  statsmodels, yfinance, jupyterlab).

---

## Instalación

```bash
cd TFM_EDA
python3.12 -m venv venv
source venv/bin/activate          # Linux / Mac
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

---

## Uso

Análisis interactivo completo:

```bash
jupyter lab            # abre y ejecuta EDA_Portfolio_TFM.ipynb
```

Regenerar solo las figuras de distribuciones, Q-Q y ACF (con las leyendas y anotaciones
claramente visibles), usando los precios cacheados en `../RL/data_cache/`:

```bash
python regenerar_figuras_leyendas.py
# Sobrescribe figuras_eda/fig6_distribuciones.png, fig7_qqplots.png, fig9_acf_cuadrados.png
```

---

## Datos

El notebook descarga los precios vía `yfinance`. El script `regenerar_figuras_leyendas.py`
lee el CSV cacheado `../RL/data_cache/portfolio_prices_20040101_20251231.csv`, por lo que no
requiere conexión a internet.
