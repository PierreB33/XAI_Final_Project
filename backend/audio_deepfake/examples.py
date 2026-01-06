"""
Script d'exemple complet montrant l'utilisation du système
"""

import os
import sys
from pathlib import Path

# Ajouter le backend au path
sys.path.insert(0, str(Path(__file__).parent))

from deepfake_detector import DeepfakeAudioDetector
from xai_explainer import XAIExplainer
from utils import ExplanationVisualizer, PredictionReport, CompatibilityChecker
from spectrogram_converter import get_audio_info
import matplotlib.pyplot as plt


def example_single_prediction():
    """Exemple 1: Prédiction simple"""
    print("\n" + "="*60)
    print("EXEMPLE 1: Prédiction Simple")
    print("="*60 + "\n")
    
    # Chemin d'exemple (remplacer par votre fichier)
    audio_path = "path/to/your/audio.wav"
    
    if not os.path.exists(audio_path):
        print(f"⚠️ Fichier audio non trouvé: {audio_path}")
        print("Créez un fichier audio .wav pour tester")
        return
    
    try:
        # Initialiser le détecteur
        detector = DeepfakeAudioDetector(model_type='mobilenet')
        
        # Faire une prédiction
        result = detector.predict(audio_path)
        
        # Afficher les résultats
        print(f"Audio: {os.path.basename(audio_path)}")
        print(f"Label: {result['predicted_label']}")
        print(f"Confiance: {result['confidence']:.2%}")
        print(f"Probabilité RÉEL: {result['real_probability']:.4f}")
        print(f"Probabilité FAUX: {result['fake_probability']:.4f}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")


def example_single_xai():
    """Exemple 2: XAI sur une prédiction"""
    print("\n" + "="*60)
    print("EXEMPLE 2: Explainabilité Grad-CAM")
    print("="*60 + "\n")
    
    audio_path = "path/to/your/audio.wav"
    
    if not os.path.exists(audio_path):
        print(f"⚠️ Fichier audio non trouvé: {audio_path}")
        return
    
    try:
        # Détection
        detector = DeepfakeAudioDetector(model_type='vgg16')
        result = detector.predict(audio_path)
        
        print(f"Prédiction: {result['predicted_label']} ({result['confidence']:.2%})")
        
        # Explainabilité
        explainer = XAIExplainer(detector.model, model_type='vgg16')
        grad_cam = explainer.grad_cam(audio_path, class_index=1)
        
        print("✓ Grad-CAM généré")
        
        # Visualizer
        fig = ExplanationVisualizer.plot_grad_cam(grad_cam)
        plt.savefig('grad_cam_result.png', dpi=150, bbox_inches='tight')
        print("✓ Graphique sauvegardé: grad_cam_result.png")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")


def example_compare_methods():
    """Exemple 3: Comparaison des méthodes XAI"""
    print("\n" + "="*60)
    print("EXEMPLE 3: Comparaison des Méthodes XAI")
    print("="*60 + "\n")
    
    audio_path = "path/to/your/audio.wav"
    
    if not os.path.exists(audio_path):
        print(f"⚠️ Fichier audio non trouvé: {audio_path}")
        return
    
    try:
        # Détection
        detector = DeepfakeAudioDetector(model_type='mobilenet')
        pred_result = detector.predict(audio_path)
        
        print(f"Prédiction: {pred_result['predicted_label']}")
        print("\nGénération des explications...")
        
        # Explainabilité
        explainer = XAIExplainer(detector.model, model_type='mobilenet')
        
        # Grad-CAM
        print("  - Grad-CAM...", end='', flush=True)
        try:
            grad_cam = explainer.grad_cam(audio_path)
            print(" ✓")
        except Exception as e:
            print(f" ✗ ({e})")
            grad_cam = None
        
        # LIME
        print("  - LIME...", end='', flush=True)
        try:
            lime = explainer.lime_explanation(audio_path, num_samples=500)
            print(" ✓")
        except Exception as e:
            print(f" ✗ ({e})")
            lime = None
        
        # SHAP
        print("  - SHAP...", end='', flush=True)
        try:
            shap = explainer.shap_explanation(audio_path)
            print(" ✓")
        except Exception as e:
            print(f" ✗ ({e})")
            shap = None
        
        # Générer un rapport
        print("\nGénération du rapport...")
        report = PredictionReport.generate_report(pred_result)
        print(report)
        
        # Sauvegarder le rapport
        PredictionReport.save_json_report(pred_result, 'prediction_report.json')
        print("✓ Rapport JSON sauvegardé: prediction_report.json")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")


def example_batch_processing():
    """Exemple 4: Traitement par batch"""
    print("\n" + "="*60)
    print("EXEMPLE 4: Traitement par Batch")
    print("="*60 + "\n")
    
    audio_files = [
        "path/to/audio1.wav",
        "path/to/audio2.wav",
        "path/to/audio3.wav",
    ]
    
    # Vérifier si les fichiers existent
    existing_files = [f for f in audio_files if os.path.exists(f)]
    
    if not existing_files:
        print("⚠️ Aucun fichier audio trouvé")
        print("Créez des fichiers .wav dans les chemins spécifiés pour tester")
        return
    
    try:
        detector = DeepfakeAudioDetector(model_type='mobilenet')
        
        print(f"Traitement de {len(existing_files)} fichiers...\n")
        
        results = detector.batch_predict(existing_files)
        
        # Statistiques
        real_count = sum(1 for r in results if r.get('predicted_label') == 'REAL')
        fake_count = sum(1 for r in results if r.get('predicted_label') == 'FAKE')
        
        print(f"\nRésultats:")
        print(f"  RÉEL: {real_count}")
        print(f"  FAUX: {fake_count}")
        print(f"  Total: {len(results)}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")


def example_audio_info():
    """Exemple 5: Informations sur un fichier audio"""
    print("\n" + "="*60)
    print("EXEMPLE 5: Informations Audio")
    print("="*60 + "\n")
    
    audio_path = "path/to/your/audio.wav"
    
    if not os.path.exists(audio_path):
        print(f"⚠️ Fichier audio non trouvé: {audio_path}")
        return
    
    try:
        info = get_audio_info(audio_path)
        
        print(f"Fichier: {os.path.basename(audio_path)}")
        print(f"Durée: {info['duration']:.2f} secondes")
        print(f"Taux d'échantillonnage: {info['sample_rate']} Hz")
        print(f"Nombre d'échantillons: {info['num_samples']}")
        print(f"Taille du fichier: {info['file_size'] / 1024:.2f} KB")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")


def example_compatibility():
    """Exemple 6: Vérification de compatibilité"""
    print("\n" + "="*60)
    print("EXEMPLE 6: Vérification de Compatibilité")
    print("="*60 + "\n")
    
    # Méthodes disponibles
    methods = ['grad_cam', 'lime', 'shap']
    
    # Vérifier la compatibilité
    print("Vérification de compatibilité pour AUDIO:\n")
    compatible, incompatible = CompatibilityChecker.filter_methods(methods, 'audio')
    
    print(f"  Compatible: {compatible}")
    print(f"  Incompatible: {incompatible}")
    
    # Afficher tous les modèles disponibles
    from audio_models import get_available_models
    print(f"\nModèles disponibles: {get_available_models()}")


def main():
    """Fonction principale"""
    print("\n" + "="*60)
    print("SYSTÈME DE DÉTECTION DEEPFAKE AUDIO - EXEMPLES")
    print("="*60)
    
    print("\nExemples disponibles:")
    print("  1. Prédiction simple")
    print("  2. Explainabilité Grad-CAM")
    print("  3. Comparaison des méthodes XAI")
    print("  4. Traitement par batch")
    print("  5. Informations audio")
    print("  6. Vérification de compatibilité")
    
    print("\nPour utiliser les exemples:")
    print("  - Modifiez les chemins des fichiers audio")
    print("  - Téléchargez les poids pré-entraînés")
    print("  - Exécutez les fonctions dans ce script")
    
    # Décommenter pour exécuter les exemples
    # example_single_prediction()
    # example_single_xai()
    # example_compare_methods()
    # example_batch_processing()
    # example_audio_info()
    # example_compatibility()


if __name__ == '__main__':
    main()
