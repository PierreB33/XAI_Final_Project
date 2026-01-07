#!/usr/bin/env python3
"""Test script pour vérifier les explications XAI"""

import requests
import sys
from pathlib import Path

# Trouver un fichier audio de test
test_audio = Path("backend/audio_deepfake/examples")

if not test_audio.exists():
    print("❌ Dossier examples non trouvé")
    sys.exit(1)

audio_files = list(test_audio.glob("*.wav"))
if not audio_files:
    print("❌ Aucun fichier WAV trouvé")
    sys.exit(1)

test_file = audio_files[0]
print(f"[TEST] Utilisant: {test_file}")

# Tester l'API
url = "http://localhost:5000/api/analyze/audio"
files = {'file': open(test_file, 'rb')}
data = {
    'model': 'deepfake-mobilenet',
    'xai_methods': ['lime', 'gradcam', 'shap']
}

print("[TEST] Envoi de la requête...")
response = requests.post(url, files=files, data=data)
print(f"[TEST] Status: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print(f"[TEST] Prédiction: {result['prediction']} ({result['confidence']*100:.1f}%)")
    
    xai_results = result.get('xai_results', {})
    print(f"[TEST] Résultats XAI: {list(xai_results.keys())}")
    
    for method, data in xai_results.items():
        if 'image' in data:
            print(f"[TEST] {method}: Image (size={len(data.get('image', ''))} chars)")
        elif 'error' in data:
            print(f"[TEST] {method}: ERROR - {data['error']}")
        else:
            print(f"[TEST] {method}: OK")
else:
    print(f"[TEST] Erreur: {response.text}")
