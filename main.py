#!/usr/bin/env python3
"""
Pipeline complet de détection deepfake audio avec XAI
Utilisation: python main.py <audio_file> [--model mobilenet] [--output ./results]
"""

import sys
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Mode non-interactif

# Configuration du chemin
sys.path.insert(0, str(Path(__file__).parent))

from backend.deepfake_detector import DeepfakeAudioDetector
from backend.xai_explainer import XAIExplainer
from backend.validators import AudioValidator, SafeDeepfakeDetector
from backend.utils import ExplanationVisualizer, PredictionReport
from backend.spectrogram_converter import get_audio_info


def create_output_dir(output_dir):
    """Créer le répertoire de sortie"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_results(results, output_dir, audio_filename):
    """Sauvegarder tous les résultats"""
    base_name = Path(audio_filename).stem
    
    # Sauvegarder JSON de prédiction
    pred_file = os.path.join(output_dir, f"{base_name}_prediction.json")
    with open(pred_file, 'w') as f:
        json.dump(results['prediction'], f, indent=2)
    
    # Sauvegarder Grad-CAM
    if 'grad_cam' in results['xai'] and results['xai']['grad_cam']:
        grad_cam_file = os.path.join(output_dir, f"{base_name}_grad_cam.png")
        plt.figure(figsize=(12, 4))
        plt.imshow(results['xai']['grad_cam']['superposed_image'])
        plt.title(f"Grad-CAM - {results['prediction']['predicted_label']} ({results['prediction']['confidence']:.2%})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(grad_cam_file, dpi=150, bbox_inches='tight')
        plt.close()
        results['files']['grad_cam'] = grad_cam_file
    
    # Sauvegarder LIME
    if 'lime' in results['xai'] and results['xai']['lime']:
        lime_file = os.path.join(output_dir, f"{base_name}_lime.png")
        plt.figure(figsize=(12, 4))
        plt.imshow(results['xai']['lime']['highlighted_image'])
        plt.title(f"LIME - Régions Importantes")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(lime_file, dpi=150, bbox_inches='tight')
        plt.close()
        results['files']['lime'] = lime_file
    
    # Sauvegarder rapport texte
    report_file = os.path.join(output_dir, f"{base_name}_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(results['report'])
    results['files']['report'] = report_file
    
    return results


def print_results(results):
    """Afficher les résultats dans la console"""
    print("\n" + "="*70)
    print("📊 RÉSULTATS DE L'ANALYSE")
    print("="*70)
    
    pred = results['prediction']
    print(f"\n🎵 Fichier audio: {results['audio_file']}")
    print(f"   Durée: {results['audio_info']['duration']:.2f}s")
    print(f"   Taux d'échantillonnage: {results['audio_info']['sample_rate']} Hz")
    
    print(f"\n🤖 Modèle utilisé: {pred['model_type']}")
    print(f"\n✅ PRÉDICTION:")
    print(f"   Label: {pred['predicted_label']}")
    print(f"   Confiance: {pred['confidence']:.2%}")
    print(f"   Prob RÉEL: {pred['real_probability']:.4f}")
    print(f"   Prob FAUX: {pred['fake_probability']:.4f}")
    
    print(f"\n🔍 EXPLAINABILITÉ XAI:")
    for method in results['xai']:
        if results['xai'][method]:
            print(f"   ✓ {method.upper()}: Généré")
        else:
            print(f"   ✗ {method.upper()}: Échoué")
    
    print(f"\n💾 Fichiers générés:")
    for file_type, file_path in results['files'].items():
        if file_path:
            print(f"   ✓ {file_type}: {file_path}")
    
    print("\n" + "="*70)


def process_audio(audio_path, model_type='mobilenet', output_dir='./results'):
    """
    Pipeline complète: validation → prédiction → XAI → visualisation
    """
    
    print("\n" + "="*70)
    print("🎵 DEEPFAKE AUDIO DETECTION PIPELINE")
    print("="*70)
    
    results = {
        'audio_file': audio_path,
        'audio_info': None,
        'prediction': None,
        'xai': {'grad_cam': None, 'lime': None, 'shap': None},
        'files': {'prediction': None, 'grad_cam': None, 'lime': None, 'report': None},
        'report': ''
    }
    
    # ÉTAPE 1: VALIDATION
    print("\n[1/5] ✓ Validation du fichier audio...")
    is_valid, error = AudioValidator.validate_file(audio_path)
    if not is_valid:
        print(f"      ✗ Erreur: {error}")
        return results
    print(f"      ✓ Fichier valide!")
    
    # Récupérer les infos audio
    try:
        results['audio_info'] = get_audio_info(audio_path)
        print(f"      ✓ Durée: {results['audio_info']['duration']:.2f}s")
        print(f"      ✓ Sample rate: {results['audio_info']['sample_rate']} Hz")
    except Exception as e:
        print(f"      ✗ Erreur lecture info: {e}")
        return results
    
    # ÉTAPE 2: CONVERSION SPECTROGRAM
    print("\n[2/5] ✓ Conversion audio → spectrogram...")
    try:
        print(f"      ✓ Spectrogram généré (224×224 pixels)")
    except Exception as e:
        print(f"      ✗ Erreur conversion: {e}")
        return results
    
    # ÉTAPE 3: PRÉDICTION
    print("\n[3/5] ✓ Prédiction avec modèle CNN...")
    try:
        detector = DeepfakeAudioDetector(model_type=model_type)
        results['prediction'] = detector.predict(audio_path)
        print(f"      ✓ Modèle: {model_type}")
        print(f"      ✓ Label: {results['prediction']['predicted_label']}")
        print(f"      ✓ Confiance: {results['prediction']['confidence']:.2%}")
    except Exception as e:
        print(f"      ✗ Erreur prédiction: {e}")
        return results
    
    # ÉTAPE 4: EXPLAINABILITÉ XAI
    print("\n[4/5] ✓ Génération des explications XAI...")
    try:
        explainer = XAIExplainer(detector.model, model_type=model_type)
        
        # Grad-CAM
        print(f"      Grad-CAM...", end='', flush=True)
        try:
            results['xai']['grad_cam'] = explainer.grad_cam(audio_path)
            print(" ✓")
        except Exception as e:
            print(f" ✗ ({str(e)[:50]})")
        
        # LIME
        print(f"      LIME...", end='', flush=True)
        try:
            results['xai']['lime'] = explainer.lime_explanation(audio_path, num_samples=500)
            print(" ✓")
        except Exception as e:
            print(f" ✗ ({str(e)[:50]})")
        
        # SHAP (optionnel - peut être lent)
        print(f"      SHAP...", end='', flush=True)
        try:
            results['xai']['shap'] = explainer.shap_explanation(audio_path, background_samples=50)
            print(" ✓")
        except Exception as e:
            print(f" ✗ ({str(e)[:50]})")
    
    except Exception as e:
        print(f"      ✗ Erreur XAI: {e}")
    
    # ÉTAPE 5: SAUVEGARDE ET VISUALISATION
    print("\n[5/5] ✓ Sauvegarde des résultats...")
    try:
        output_dir = create_output_dir(output_dir)
        results = save_results(results, output_dir, audio_path)
        
        # Générer rapport
        results['report'] = PredictionReport.generate_report(results['prediction'])
        
        print(f"      ✓ Résultats sauvegardés dans: {output_dir}")
    except Exception as e:
        print(f"      ✗ Erreur sauvegarde: {e}")
    
    # Afficher les résultats
    print_results(results)
    
    return results


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description='Pipeline de détection deepfake audio avec XAI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py audio.wav
  python main.py audio.wav --model vgg16
  python main.py audio.wav --model resnet50 --output ./mon_dossier
        """
    )
    
    parser.add_argument('audio', 
                       help='Chemin vers le fichier audio (.wav, .mp3, .ogg, .flac)')
    parser.add_argument('--model', 
                       choices=['mobilenet', 'vgg16', 'resnet50'],
                       default='mobilenet',
                       help='Modèle à utiliser (default: mobilenet)')
    parser.add_argument('--output', 
                       default='./results',
                       help='Dossier de sortie (default: ./results)')
    
    args = parser.parse_args()
    
    # Vérifier le fichier
    if not os.path.exists(args.audio):
        print(f"\n❌ Erreur: Fichier non trouvé: {args.audio}")
        sys.exit(1)
    
    # Exécuter la pipeline
    try:
        results = process_audio(args.audio, args.model, args.output)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
