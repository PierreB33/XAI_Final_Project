"""
Fichier principal pour tester les fonctionnalités de détection de deepfake audio
"""

import os
import sys
from pathlib import Path

# Ajouter le répertoire backend au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepfake_detector import DeepfakeAudioDetector
from xai_explainer import XAIExplainer
from audio_models import get_available_models
from config import CLASS_MAPPING, AUDIO_MODELS


def test_detection(audio_path, model_type='mobilenet', weights_path=None):
    """
    Tester la détection sur un fichier audio
    
    Args:
        audio_path (str): Chemin vers le fichier .wav
        model_type (str): Type du modèle
        weights_path (str): Chemin vers les poids entraînés
    """
    print(f"\n{'='*60}")
    print(f"Test de détection: {os.path.basename(audio_path)}")
    print(f"Modèle: {model_type}")
    print(f"{'='*60}\n")
    
    try:
        # Initialiser le détecteur
        detector = DeepfakeAudioDetector(model_type=model_type, weights_path=weights_path)
        
        # Faire une prédiction
        result = detector.predict(audio_path)
        
        # Afficher les résultats
        print("RÉSULTATS DE LA PRÉDICTION:")
        print(f"  Label prédit: {result['predicted_label']}")
        print(f"  Probabilité RÉEL: {result['real_probability']:.4f}")
        print(f"  Probabilité FAUX: {result['fake_probability']:.4f}")
        print(f"  Confiance: {result['confidence']:.4f}")
        print(f"  Modèle utilisé: {result['model_type']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return None


def test_xai(audio_path, model, model_type='mobilenet', method='all'):
    """
    Tester les méthodes XAI
    
    Args:
        audio_path (str): Chemin vers le fichier .wav
        model: Modèle TensorFlow
        model_type (str): Type du modèle
        method (str): 'grad_cam', 'lime', 'shap', ou 'all'
    """
    print(f"\n{'='*60}")
    print(f"Test XAI: {os.path.basename(audio_path)}")
    print(f"Méthodes: {method}")
    print(f"{'='*60}\n")
    
    try:
        explainer = XAIExplainer(model, model_type=model_type)
        
        if method == 'all':
            results = explainer.compare_methods(audio_path)
            print("✓ Explications générées pour:")
            for method_name in results['methods'].keys():
                if 'error' not in results['methods'][method_name]:
                    print(f"  - {method_name}")
                else:
                    print(f"  - {method_name} (❌ Erreur)")
        else:
            if method == 'grad_cam':
                result = explainer.grad_cam(audio_path)
            elif method == 'lime':
                result = explainer.lime_explanation(audio_path)
            elif method == 'shap':
                result = explainer.shap_explanation(audio_path)
            
            print(f"✓ Explication {method} générée")
        
        return results if method == 'all' else result
        
    except Exception as e:
        print(f"❌ Erreur XAI: {str(e)}")
        return None


def main():
    """Fonction principale pour les tests"""
    print("\n" + "="*60)
    print("SYSTÈME DE DÉTECTION DE DEEPFAKE AUDIO")
    print("="*60)
    
    # Afficher les modèles disponibles
    print("\n📊 Modèles disponibles:", ', '.join(get_available_models()))
    print("🔍 Méthodes XAI: grad_cam, lime, shap")
    
    # Exemple d'utilisation (à adapter avec vos fichiers audio)
    print("\n" + "="*60)
    print("EXEMPLE D'UTILISATION")
    print("="*60)
    
    print("""
    from deepfake_detector import DeepfakeAudioDetector
    from xai_explainer import XAIExplainer
    
    # Détection
    detector = DeepfakeAudioDetector(model_type='mobilenet')
    result = detector.predict('path/to/audio.wav')
    
    # XAI
    explainer = XAIExplainer(detector.model)
    grad_cam = explainer.grad_cam('path/to/audio.wav')
    lime_exp = explainer.lime_explanation('path/to/audio.wav')
    shap_exp = explainer.shap_explanation('path/to/audio.wav')
    """)


if __name__ == '__main__':
    main()
