"""
Utilitaires XAI supplémentaires
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class GradCAMVisualizer:
    """Classe pour visualiser et analyser les résultats Grad-CAM"""
    
    @staticmethod
    def create_overlay(original_image, heatmap, alpha=0.5):
        """
        Créer une superposition entre l'image originale et la heatmap
        
        Args:
            original_image: Image originale
            heatmap: Heatmap générée
            alpha: Transparence de la heatmap
            
        Returns:
            Image superposée
        """
        # Normaliser la heatmap entre 0 et 1
        heatmap_normalized = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        
        # Créer l'image avec couleur jet
        heatmap_colored = plt.cm.jet(heatmap_normalized)
        
        # Superposer
        overlay = alpha * heatmap_colored[:, :, :3] + (1 - alpha) * (original_image / 255.0)
        
        return (overlay * 255).astype(np.uint8)


class ExplanationAnalyzer:
    """Analyser et comparer les explications"""
    
    @staticmethod
    def compare_xai_results(results_dict: Dict) -> Dict:
        """
        Analyser et comparer les résultats de plusieurs méthodes XAI
        
        Args:
            results_dict: Dictionnaire avec les résultats de chaque méthode
            
        Returns:
            Analyse comparative
        """
        analysis = {
            'methods_used': [],
            'methods_failed': [],
            'consistency_score': 0.0
        }
        
        for method, result in results_dict.items():
            if 'error' not in result:
                analysis['methods_used'].append(method)
            else:
                analysis['methods_failed'].append(method)
        
        return analysis
    
    @staticmethod
    def calculate_explanation_quality(explanation_dict: Dict) -> float:
        """
        Calculer une métrique de qualité pour une explication
        
        Args:
            explanation_dict: Résultat d'une explication
            
        Returns:
            Score de qualité (0-1)
        """
        # Cette métrique pourrait être basée sur plusieurs critères
        # Pour l'instant, retourner 0.5 comme placeholder
        return 0.5


class InteractiveExplainer:
    """Classe pour les explications interactives"""
    
    @staticmethod
    def create_detailed_report(prediction_result: Dict, xai_results: Dict) -> str:
        """
        Créer un rapport détaillé combinant prédiction et XAI
        
        Args:
            prediction_result: Résultat de la prédiction
            xai_results: Résultats des explications XAI
            
        Returns:
            Rapport texte détaillé
        """
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║          RAPPORT D'ANALYSE DEEPFAKE AUDIO DÉTAILLÉ           ║
╚══════════════════════════════════════════════════════════════╝

📊 RÉSULTAT DE LA PRÉDICTION:
  • Label: {prediction_result.get('predicted_label', 'N/A')}
  • Confiance: {prediction_result.get('confidence', 0):.2%}
  • Probabilité RÉEL: {prediction_result.get('real_probability', 0):.4f}
  • Probabilité FAUX: {prediction_result.get('fake_probability', 0):.4f}
  • Modèle: {prediction_result.get('model_type', 'N/A')}

🔍 EXPLICATIONS XAI:
"""
        
        if xai_results:
            for method, result in xai_results.items():
                if 'error' not in result:
                    report += f"  ✓ {method.upper()}: Succès\n"
                else:
                    report += f"  ✗ {method.upper()}: Échoué - {result['error']}\n"
        
        report += "\n╔══════════════════════════════════════════════════════════════╗\n"
        
        return report
