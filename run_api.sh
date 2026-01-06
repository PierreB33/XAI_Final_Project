#!/bin/bash
# Script pour lancer l'API FastAPI

echo "🚀 Démarrage de l'API XAI Platform..."
echo "Installation des dépendances..."
pip install fastapi uvicorn -q

echo "Lancement du serveur sur http://localhost:5000"
echo "Documentation Swagger: http://localhost:5000/docs"
echo ""

python api.py
