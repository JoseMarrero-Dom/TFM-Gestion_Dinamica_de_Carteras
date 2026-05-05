# TFM — Setup del entorno

## Requisitos

- Python 3.12
- Git

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/JoseMarrero-Dom/tfm.git
cd tfm

# 2. Crear el entorno virtual
cd TFM_EDA
python3 -m venv venv

# 3. Activar el entorno
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows

# 4. Instalar dependencias
pip install -r requirements.txt
```

## Uso

Con el entorno activado, lanza JupyterLab:

```bash
jupyter lab
```

Abre el notebook `EDA_Portfolio_TFM.ipynb`.

## Estructura

```
TFM/
├── TFM_EDA/
│   ├── EDA_Portfolio_TFM.ipynb   # Notebook principal
│   ├── figuras_eda/              # Gráficas y tablas generadas
│   └── requirements.txt          # Dependencias del entorno
└── README.md
```
