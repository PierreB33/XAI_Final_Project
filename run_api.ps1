# Script pour lancer l'API FastAPI sur Windows

Write-Host "🚀 Démarrage de l'API XAI Platform..."
Write-Host "Installation des dépendances..."
pip install fastapi uvicorn -q

Write-Host "Lancement du serveur sur http://localhost:5000"
Write-Host "Documentation Swagger: http://localhost:5000/docs"
Write-Host ""

python api.py
