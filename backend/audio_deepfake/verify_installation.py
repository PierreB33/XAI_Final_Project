#!/usr/bin/env python3
"""
Script de vérification : Teste que tous les imports et structures fonctionnent
Exécutez: python backend/audio_deepfake/verify_installation.py
"""

import sys
import os
from pathlib import Path

# Ajouter le backend au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def check_imports():
    """Vérifier que tous les imports fonctionnent"""
    print("🔍 Vérification des imports...\n")
    
    checks = [
        ("TensorFlow", "import tensorflow as tf"),
        ("Keras", "from tensorflow.keras import layers"),
        ("LibROSA", "import librosa"),
        ("LIME", "from lime.image import ImageExplainer"),
        ("SHAP", "import shap"),
        ("NumPy", "import numpy as np"),
        ("OpenCV", "import cv2"),
        ("Pillow", "from PIL import Image"),
        ("Matplotlib", "import matplotlib.pyplot as plt"),
    ]
    
    failed = []
    for name, import_stmt in checks:
        try:
            exec(import_stmt)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {str(e)}")
            failed.append(name)
    
    return len(failed) == 0, failed


def check_backend_modules():
    """Vérifier que les modules backend peuvent être importés"""
    print("\n🔍 Vérification des modules backend...\n")
    
    modules = [
        ("audio_models", "build_mobilenet_model"),
        ("deepfake_detector", "DeepfakeAudioDetector"),
        ("xai_explainer", "XAIExplainer"),
        ("spectrogram_converter", "create_spectrogram"),
        ("config", "AUDIO_MODELS"),
        ("utils", "ExplanationVisualizer"),
        ("validators", "AudioValidator"),
    ]
    
    failed = []
    for module_name, class_or_func in modules:
        try:
            module = __import__(f"backend.audio_deepfake.{module_name}", fromlist=[class_or_func])
            getattr(module, class_or_func)
            print(f"  ✅ backend.audio_deepfake.{module_name}.{class_or_func}")
        except (ImportError, AttributeError) as e:
            print(f"  ❌ backend.audio_deepfake.{module_name}.{class_or_func}: {str(e)}")
            failed.append(f"{module_name}.{class_or_func}")
    
    return len(failed) == 0, failed


def check_configuration():
    """Vérifier la configuration"""
    print("\n🔍 Vérification de la configuration...\n")
    
    try:
        from backend.audio_deepfake.config import (
            AUDIO_MODELS, XAI_METHODS, CLASS_MAPPING, 
            MODEL_INPUT, AUDIO_PREPROCESSING
        )
        
        print(f"  ✅ Modèles audio: {AUDIO_MODELS}")
        print(f"  ✅ Méthodes XAI: {XAI_METHODS}")
        print(f"  ✅ Classes: {list(CLASS_MAPPING.values())}")
        print(f"  ✅ Taille entrée: {MODEL_INPUT['image_size']}")
        
        return True, []
    except Exception as e:
        print(f"  ❌ Configuration: {str(e)}")
        return False, [str(e)]


def check_model_creation():
    """Tester la création des modèles"""
    print("\n🔍 Vérification de la création des modèles...\n")
    
    try:
        from backend.audio_deepfake.audio_models import (
            build_mobilenet_model, 
            build_vgg16_model, 
            build_resnet_model
        )
        
        models = [
            ("MobileNet", build_mobilenet_model),
            ("VGG16", build_vgg16_model),
            ("ResNet50", build_resnet_model),
        ]
        
        failed = []
        for name, builder in models:
            try:
                model = builder()
                print(f"  ✅ {name} ({model.count_params():,} params)")
            except Exception as e:
                print(f"  ❌ {name}: {str(e)}")
                failed.append(name)
        
        return len(failed) == 0, failed
    except Exception as e:
        print(f"  ❌ Model creation: {str(e)}")
        return False, [str(e)]


def main():
    """Exécuter toutes les vérifications"""
    print("="*60)
    print("VÉRIFICATION D'INSTALLATION - BACKEND DEEPFAKE AUDIO")
    print("="*60 + "\n")
    
    results = []
    
    # Vérifications
    success, failed = check_imports()
    results.append(("Dépendances Python", success, failed))
    
    success, failed = check_backend_modules()
    results.append(("Modules backend", success, failed))
    
    success, failed = check_configuration()
    results.append(("Configuration", success, failed))
    
    success, failed = check_model_creation()
    results.append(("Création modèles", success, failed))
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60 + "\n")
    
    all_passed = True
    for name, success, failed in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
        if failed:
            all_passed = False
            for item in failed:
                print(f"       ⚠️  {item}")
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ TOUTES LES VÉRIFICATIONS SONT PASSÉES!")
        print("\nProchaines étapes:")
        print("  1. Télécharger les poids pré-entraînés")
        print("  2. Remplir les chemins dans backend/audio_deepfake/config.py")
        print("  3. Tester avec un fichier audio: python backend/audio_deepfake/test_audio.py")
        return 0
    else:
        print("❌ CERTAINES VÉRIFICATIONS ONT ÉCHOUÉ")
        print("\nRéinstallez les dépendances:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
