#!/bin/bash

# 🚀 Script para crear proyectos Python aislados
# ./crear_proyecto.sh <nombre_proyecto>
# permisos: chmod +x crear_proyecto.sh
if [ -z "$1" ]; then
    echo "Uso: ./crear_proyecto.sh nombre_proyecto"
    exit 1
fi

PROJECT_NAME=$1

echo "📁 Creando carpeta del proyecto..."
mkdir $PROJECT_NAME
cd $PROJECT_NAME

echo "🐍 Creando entorno virtual..."
python3 -m venv venv

echo "🔛 Activando entorno..."
source venv/bin/activate

echo "📦 Instalando dependencias básicas..."
pip install --upgrade pip
pip install jupyter ipykernel

echo "🧠 Creando kernel de Jupyter..."
python -m ipykernel install --user --name "$PROJECT_NAME" --display-name "Python ($PROJECT_NAME)"

echo "🎉 Proyecto creado!"
echo "Ruta: $(pwd)"
