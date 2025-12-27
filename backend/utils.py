"""
Utilitaires pour le système de détection et explainabilité
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json


class ExplanationVisualizer:
    """Classe pour visualiser les explications XAI"""
    
    @staticmethod
    def plot_grad_cam(grad_cam_result, title='Grad-CAM Visualization'):
        """
        Visualiser le résultat Grad-CAM
        
        Args:
            grad_cam_result (dict): Résultat de grad_cam()
            title (str): Titre du graphique
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Heatmap
        axes[0].imshow(grad_cam_result['heatmap'], cmap='jet')
        axes[0].set_title('Heatmap')
        axes[0].axis('off')
        
        # Image superposée
        axes[1].imshow(grad_cam_result['superposed_image'])
        axes[1].set_title(title)
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_lime(lime_result, title='LIME Explanation'):
        """
        Visualiser le résultat LIME
        
        Args:
            lime_result (dict): Résultat de lime_explanation()
            title (str): Titre du graphique
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Image originale
        axes[0].imshow(lime_result['highlighted_image'])
        axes[0].set_title(title)
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(lime_result['mask'], cmap='RdYlGn')
        axes[1].set_title('Feature Importance')
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def compare_explanations(methods_dict):
        """
        Comparer les explications de plusieurs méthodes
        
        Args:
            methods_dict (dict): Dictionnaire avec les résultats de chaque méthode
        """
        num_methods = len([m for m in methods_dict.keys() if 'error' not in methods_dict[m]])
        fig, axes = plt.subplots(1, num_methods, figsize=(15, 4))
        
        if num_methods == 1:
            axes = [axes]
        
        idx = 0
        for method_name, result in methods_dict.items():
            if 'error' in result:
                continue
            
            if method_name == 'grad_cam':
                axes[idx].imshow(result['superposed_image'])
                axes[idx].set_title('Grad-CAM')
            elif method_name == 'lime':
                axes[idx].imshow(result['highlighted_image'])
                axes[idx].set_title('LIME')
            elif method_name == 'shap':
                # SHAP values visualization would be more complex
                axes[idx].text(0.5, 0.5, 'SHAP\n(Complex visualization)', 
                             ha='center', va='center', fontsize=12)
                axes[idx].set_title('SHAP')
            
            axes[idx].axis('off')
            idx += 1
        
        plt.tight_layout()
        return fig


class PredictionReport:
    """Classe pour générer des rapports de prédiction"""
    
    @staticmethod
    def generate_report(detection_result, xai_results=None, output_path=None):
        """
        Générer un rapport complet
        
        Args:
            detection_result (dict): Résultat de detector.predict()
            xai_results (dict): Résultats XAI (optionnel)
            output_path (str): Chemin pour sauvegarder le rapport
            
        Returns:
            str: Texte du rapport
        """
        report = f"""
{'='*60}
RAPPORT D'ANALYSE DEEPFAKE AUDIO
{'='*60}

INFORMATIONS:
  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Audio: {detection_result.get('audio_path', 'N/A')}

RÉSULTATS DE DÉTECTION:
  Modèle utilisé: {detection_result['model_type']}
  Label prédit: {detection_result['predicted_label']}
  Probabilité RÉEL: {detection_result['real_probability']:.4f}
  Probabilité FAUX: {detection_result['fake_probability']:.4f}
  Confiance: {detection_result['confidence']:.4f}

"""
        
        if xai_results:
            report += f"""EXPLICATIONS XAI:
"""
            for method, result in xai_results.items():
                if 'error' not in result:
                    report += f"  ✓ {method.upper()}: Généré avec succès\n"
                else:
                    report += f"  ✗ {method.upper()}: {result['error']}\n"
        
        report += f"\n{'='*60}\n"
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    @staticmethod
    def save_json_report(detection_result, output_path):
        """
        Sauvegarder le résultat en JSON
        
        Args:
            detection_result (dict): Résultat de detector.predict()
            output_path (str): Chemin du fichier JSON
        """
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'detection': detection_result,
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)


class CompatibilityChecker:
    """Vérifier la compatibilité des méthodes XAI avec les types d'entrée"""
    
    @staticmethod
    def get_compatible_methods(input_type='audio'):
        """
        Obtenir les méthodes XAI compatibles avec le type d'entrée
        
        Args:
            input_type (str): 'audio' ou 'image'
            
        Returns:
            list: Méthodes compatibles
        """
        from config import XAI_COMPATIBILITY
        return XAI_COMPATIBILITY.get(input_type, [])
    
    @staticmethod
    def is_method_compatible(method, input_type='audio'):
        """
        Vérifier si une méthode est compatible avec le type d'entrée
        
        Args:
            method (str): Nom de la méthode XAI
            input_type (str): 'audio' ou 'image'
            
        Returns:
            bool: True si compatible
        """
        compatible = CompatibilityChecker.get_compatible_methods(input_type)
        return method in compatible
    
    @staticmethod
    def filter_methods(methods, input_type='audio'):
        """
        Filtrer les méthodes pour garder seulement les compatibles
        
        Args:
            methods (list): Liste des méthodes à filtrer
            input_type (str): 'audio' ou 'image'
            
        Returns:
            tuple: (méthodes compatibles, méthodes incompatibles)
        """
        compatible = []
        incompatible = []
        
        for method in methods:
            if CompatibilityChecker.is_method_compatible(method, input_type):
                compatible.append(method)
            else:
                incompatible.append(method)
        
        return compatible, incompatible


class PerformanceMetrics:
    """Classe pour calculer les métriques de performance"""
    
    @staticmethod
    def calculate_confidence_score(pred_probabilities):
        """
        Calculer un score de confiance
        
        Args:
            pred_probabilities (dict): Dictionnaire avec les probabilités
            
        Returns:
            dict: Score de confiance et niveau
        """
        max_prob = max(pred_probabilities.values())
        
        if max_prob >= 0.85:
            level = 'HIGH'
        elif max_prob >= 0.65:
            level = 'MEDIUM'
        else:
            level = 'LOW'
        
        return {
            'score': max_prob,
            'level': level
        }
    
    @staticmethod
    def calculate_agreement(results_list):
        """
        Calculer l'accord entre plusieurs prédictions
        
        Args:
            results_list (list): Liste des résultats de prédiction
            
        Returns:
            float: Taux d'accord
        """
        if len(results_list) < 2:
            return 1.0
        
        labels = [r['predicted_label'] for r in results_list]
        agreement = sum(1 for l in labels if l == labels[0]) / len(labels)
        
        return agreement
