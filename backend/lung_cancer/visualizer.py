"""
Lung Cancer Detection Backend - Visualization
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datetime import datetime

from .utils import ImageUtils, VisualizationUtils, FileUtils


class ResultVisualizer:
    """
    Classe pour visualiser et sauvegarder les résultats
    """
    
    def __init__(self, output_dir: str = './results'):
        """
        Initialiser le visualiseur
        
        Args:
            output_dir (str): Répertoire de sortie
        """
        self.output_dir = FileUtils.ensure_directory(output_dir)
    
    def save_results(self, image_path: str, prediction_result: Dict,
                    xai_result: Dict) -> Dict:
        """
        Sauvegarder tous les résultats (images + visualisations)
        
        Args:
            image_path (str): Chemin vers l'image originale
            prediction_result (dict): Résultat de la prédiction
            xai_result (dict): Résultats XAI
            
        Returns:
            dict: Chemins vers les images sauvegardées
        """
        # Créer un sous-dossier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = FileUtils.join_path(self.output_dir, timestamp)
        FileUtils.ensure_directory(result_dir)
        
        saved_files = {}
        
        # Charger l'image originale
        original_image = ImageUtils.load_cv2_image(image_path)
        
        # Sauvegarder l'image originale avec le label de prédiction
        original_with_label = ImageUtils.overlay_text(
            original_image,
            f"Prediction: {prediction_result['predicted_class']} ({prediction_result['confidence']:.2%})"
        )
        original_path = FileUtils.join_path(result_dir, '1_original_with_label.png')
        ImageUtils.save_image(original_with_label, original_path)
        saved_files['original_with_label'] = original_path
        
        # Sauvegarder Grad-CAM
        if 'gradcam' in xai_result:
            gradcam_path = FileUtils.join_path(result_dir, '2_gradcam_overlay.png')
            ImageUtils.save_image(xai_result['gradcam']['overlay'], gradcam_path)
            saved_files['gradcam'] = gradcam_path
        
        # Sauvegarder LIME
        if 'lime' in xai_result:
            lime_path = FileUtils.join_path(result_dir, '3_lime_mask.png')
            lime_image = xai_result['lime']['mask']
            ImageUtils.save_image(lime_image, lime_path)
            saved_files['lime'] = lime_path
        
        # Sauvegarder le graphique des contributions LIME
        if 'lime' in xai_result and xai_result['lime']['feature_contributions']:
            self._save_lime_contributions_chart(
                xai_result['lime']['feature_contributions'],
                FileUtils.join_path(result_dir, '4_lime_contributions.png')
            )
            saved_files['lime_chart'] = FileUtils.join_path(result_dir, '4_lime_contributions.png')
        
        # Sauvegarder un rapport texte
        report_path = FileUtils.join_path(result_dir, 'report.txt')
        self._save_report(image_path, prediction_result, report_path)
        saved_files['report'] = report_path
        
        return saved_files
    
    def _save_lime_contributions_chart(self, feature_contributions: List,
                                     output_path: str, top_n: int = 10) -> None:
        """
        Sauvegarder un graphique des contributions des features LIME
        
        Args:
            feature_contributions (list): Liste des contributions (feature_id, weight)
            output_path (str): Chemin de sauvegarde
            top_n (int): Top N features à afficher
        """
        # Trier et sélectionner les top features
        sorted_features = sorted(feature_contributions, key=lambda x: abs(x[1]), reverse=True)[:top_n]
        
        # Préparer les données
        feature_ids = [f"Feature {x[0]}" for x in sorted_features]
        weights = [x[1] for x in sorted_features]
        
        # Créer le graphique
        colors = ['green' if w > 0 else 'red' for w in weights]
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_ids, weights, color=colors)
        plt.xlabel('Contribution Weight')
        plt.ylabel('Features')
        plt.title('LIME Feature Contributions (Top 10)')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _save_report(self, image_path: str, prediction_result: Dict,
                    output_path: str) -> None:
        """
        Sauvegarder un rapport texte
        
        Args:
            image_path (str): Chemin vers l'image
            prediction_result (dict): Résultat de la prédiction
            output_path (str): Chemin de sauvegarde
        """
        report = f"""
================================================================================
LUNG CANCER DETECTION REPORT
================================================================================

Image: {image_path}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PREDICTION RESULTS:
----------------
Predicted Class: {prediction_result['predicted_class']}
Confidence: {prediction_result['confidence']:.4f} ({prediction_result['confidence']*100:.2f}%)

Class Probabilities:
"""
        
        for class_name, prob in prediction_result['probabilities'].items():
            report += f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)\n"
        
        report += f"""
EXPLANATION TECHNIQUES:
----------------
- Grad-CAM: Highlights important regions in the image
- LIME: Shows feature contributions to the prediction

================================================================================
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
    
    def create_comparison_grid(self, results: Dict, output_path: str) -> None:
        """
        Créer une grille de comparaison entre les explications
        
        Args:
            results (dict): Résultats contenant original, gradcam, lime
            output_path (str): Chemin de sauvegarde
        """
        images = {}
        titles = {}
        
        if 'original' in results:
            images['original'] = results['original']
            titles['original'] = 'Original Image'
        
        if 'gradcam' in results:
            images['gradcam'] = results['gradcam']
            titles['gradcam'] = 'Grad-CAM'
        
        if 'lime' in results:
            images['lime'] = results['lime']
            titles['lime'] = 'LIME'
        
        VisualizationUtils.create_comparison_grid(images, titles, output_path)
    
    def create_probability_chart(self, probabilities: Dict,
                                output_path: str) -> None:
        """
        Créer un graphique des probabilités
        
        Args:
            probabilities (dict): Dictionnaire {classe: probabilité}
            output_path (str): Chemin de sauvegarde
        """
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
        
        colors = ['#FF6B6B' if p == max(probs) else '#4ECDC4' for p in probs]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, probs, color=colors)
        
        plt.ylabel('Probability')
        plt.title('Lung Cancer Detection - Class Probabilities')
        plt.ylim(0, 1)
        
        # Ajouter les valeurs sur les barres
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.2%}', ha='center', va='bottom')
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()


class ReportGenerator:
    """
    Classe pour générer des rapports détaillés
    """
    
    @staticmethod
    def generate_html_report(prediction_result: Dict,
                            image_files: Dict,
                            output_path: str) -> None:
        """
        Générer un rapport HTML interactif
        
        Args:
            prediction_result (dict): Résultat de la prédiction
            image_files (dict): Chemins vers les images sauvegardées
            output_path (str): Chemin de sauvegarde du HTML
        """
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lung Cancer Detection Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1, h2 {{
            color: #333;
        }}
        .prediction {{
            font-size: 1.5em;
            color: #2ecc71;
            font-weight: bold;
            margin: 20px 0;
        }}
        .confidence {{
            font-size: 1.2em;
            color: #3498db;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        table th, table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        table th {{
            background-color: #3498db;
            color: white;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .image-section {{
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🫁 Lung Cancer Detection Report</h1>
        
        <div class="prediction">
            Predicted Class: {prediction_result['predicted_class']}
        </div>
        
        <div class="confidence">
            Confidence: {prediction_result['confidence']:.2%}
        </div>
        
        <h2>Class Probabilities</h2>
        <table>
            <tr>
                <th>Class</th>
                <th>Probability</th>
                <th>Percentage</th>
            </tr>
"""
        
        for class_name, prob in prediction_result['probabilities'].items():
            html_content += f"""
            <tr>
                <td>{class_name}</td>
                <td>{prob:.4f}</td>
                <td>{prob*100:.2f}%</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <h2>Explanations</h2>
"""
        
        if 'gradcam' in image_files:
            html_content += f"""
        <div class="image-section">
            <h3>Grad-CAM Visualization</h3>
            <p>Highlights the regions that contributed most to the prediction.</p>
            <img src="{image_files['gradcam']}" alt="Grad-CAM">
        </div>
"""
        
        if 'lime' in image_files:
            html_content += f"""
        <div class="image-section">
            <h3>LIME Explanation</h3>
            <p>Shows which features contributed positively or negatively.</p>
            <img src="{image_files['lime']}" alt="LIME">
        </div>
"""
        
        if 'lime_chart' in image_files:
            html_content += f"""
        <div class="image-section">
            <h3>Feature Contributions</h3>
            <img src="{image_files['lime_chart']}" alt="Feature Contributions">
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)
