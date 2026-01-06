"""
Lung Cancer Detection Backend - Exemple d'utilisation
"""

import os
import torch
from pathlib import Path

from .predictor import Predictor
from .xai import XAIExplainer
from .visualizer import ResultVisualizer, ReportGenerator
from .utils import DeviceUtils


def example_full_pipeline(image_path: str, model_path: str,
                         output_dir: str = './results'):
    """
    Exemple complet du pipeline de détection et explication
    
    Args:
        image_path (str): Chemin vers l'image à analyser
        model_path (str): Chemin vers les poids du modèle (.pth)
        output_dir (str): Répertoire de sortie
    """
    
    print("\n" + "="*70)
    print("🫁 LUNG CANCER DETECTION - COMPLETE PIPELINE")
    print("="*70 + "\n")
    
    # 1. Configuration
    print("📋 Configuration...")
    device = DeviceUtils.get_device(use_cuda=True)
    device_info = DeviceUtils.get_device_info()
    print(f"   Device: {device_info['current_device']}")
    
    # 2. Initialiser le prédicteur
    print("🔧 Initializing predictor...")
    predictor = Predictor(model_path=model_path, device=device)
    model_info = predictor.get_model_info()
    print(f"   Model: {model_info['model_type']}")
    print(f"   Classes: {', '.join(model_info['class_names'])}")
    
    # 3. Prédiction
    print("\n🔍 Making prediction...")
    prediction_result = predictor.predict(image_path)
    print(f"   Predicted Class: {prediction_result['predicted_class']}")
    print(f"   Confidence: {prediction_result['confidence']:.2%}")
    print(f"\n   All Probabilities:")
    for class_name, prob in prediction_result['probabilities'].items():
        print(f"      - {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    
    # 4. XAI Explainability
    print("\n🧠 Generating XAI explanations...")
    
    # Prétraiter l'image
    image_tensor = predictor.preprocessor.preprocess(image_path, device)
    
    # Créer l'explaineur
    explainer = XAIExplainer(predictor.model, device)
    
    # Générer les explications
    print("   - Generating Grad-CAM...")
    xai_results = explainer.explain_complete(
        image_path,
        image_tensor,
        prediction_result['predicted_class_idx'],
        alpha=0.5,
        num_samples=1000
    )
    print("   ✅ Grad-CAM generated")
    
    print("   - Generating LIME...")
    print("   ✅ LIME generated")
    
    # 5. Visualisation et Sauvegarde
    print("\n💾 Saving results...")
    visualizer = ResultVisualizer(output_dir=output_dir)
    
    saved_files = visualizer.save_results(
        image_path,
        prediction_result,
        xai_results
    )
    
    print(f"   Saved to: {saved_files.get('report', 'N/A')}")
    for key, path in saved_files.items():
        print(f"      - {key}: {path}")
    
    # 6. Générer rapport HTML
    print("\n📄 Generating HTML report...")
    html_path = os.path.join(os.path.dirname(saved_files['report']), 'report.html')
    ReportGenerator.generate_html_report(prediction_result, saved_files, html_path)
    print(f"   ✅ HTML report saved: {html_path}")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70 + "\n")
    
    return {
        'prediction': prediction_result,
        'xai': xai_results,
        'files': saved_files
    }


def example_simple_prediction(image_path: str, model_path: str):
    """
    Exemple simple: juste la prédiction
    
    Args:
        image_path (str): Chemin vers l'image
        model_path (str): Chemin vers les poids du modèle
    """
    print("\n🫁 Simple Prediction Example\n")
    
    device = DeviceUtils.get_device()
    predictor = Predictor(model_path=model_path, device=device)
    
    result = predictor.predict(image_path)
    
    print(f"Image: {image_path}")
    print(f"Prediction: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    return result


def example_grad_cam_only(image_path: str, model_path: str,
                         output_path: str):
    """
    Exemple: Grad-CAM seulement
    
    Args:
        image_path (str): Chemin vers l'image
        model_path (str): Chemin vers les poids du modèle
        output_path (str): Chemin de sauvegarde
    """
    print("\n🫁 Grad-CAM Explanation Example\n")
    
    device = DeviceUtils.get_device()
    
    # Prédiction
    predictor = Predictor(model_path=model_path, device=device)
    prediction_result = predictor.predict(image_path)
    
    print(f"Prediction: {prediction_result['predicted_class']}")
    
    # Grad-CAM
    image_tensor = predictor.preprocessor.preprocess(image_path, device)
    explainer = XAIExplainer(predictor.model, device)
    
    gradcam = explainer.gradcam.explain(
        image_path,
        image_tensor,
        prediction_result['predicted_class_idx']
    )
    
    # Sauvegarder
    visualizer = ResultVisualizer()
    import cv2
    cv2.imwrite(output_path, gradcam['overlay'])
    print(f"Grad-CAM saved to: {output_path}")
    
    return gradcam


def example_lime_only(image_path: str, model_path: str):
    """
    Exemple: LIME seulement
    
    Args:
        image_path (str): Chemin vers l'image
        model_path (str): Chemin vers les poids du modèle
    """
    print("\n🫁 LIME Explanation Example\n")
    
    device = DeviceUtils.get_device()
    
    # Prédiction
    predictor = Predictor(model_path=model_path, device=device)
    prediction_result = predictor.predict(image_path)
    
    print(f"Prediction: {prediction_result['predicted_class']}")
    
    # LIME
    explainer = XAIExplainer(predictor.model, device)
    lime_result = explainer.lime.explain(
        image_path,
        prediction_result['predicted_class_idx']
    )
    
    print(f"LIME explanation generated with {len(lime_result['feature_contributions'])} features")
    
    return lime_result


if __name__ == '__main__':
    # Exemples d'utilisation (à décommenter et adapter)
    
    # Model path
    model_path = 'path/to/model.pth'
    image_path = 'path/to/image.png'
    
    # Exemple 1: Pipeline complet
    # result = example_full_pipeline(image_path, model_path)
    
    # Exemple 2: Prédiction simple
    # result = example_simple_prediction(image_path, model_path)
    
    # Exemple 3: Grad-CAM seulement
    # gradcam = example_grad_cam_only(image_path, model_path, 'gradcam_output.png')
    
    # Exemple 4: LIME seulement
    # lime = example_lime_only(image_path, model_path)
    
    print("Examples are available in this module.")
    print("Uncomment the examples at the bottom to run them.")
