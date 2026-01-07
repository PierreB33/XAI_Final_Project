"""
Lung Cancer Detection Backend - XAI Explainability
Techniques: Grad-CAM, LIME, SHAP
"""

import os
import torch
import numpy as np
import cv2
from typing import Dict, Tuple
from lime import lime_image
import shap
import warnings

from .config import GRADCAM_CONFIG, LIME_CONFIG
from .utils import ImageUtils

warnings.filterwarnings('ignore')


class GradCAMExplainer:
    """
    Classe pour générer les explications Grad-CAM
    Visualise les régions de l'image qui activent le plus le modèle
    """
    
    def __init__(self, model, target_layer_name: str = 'layer4'):
        """
        Initialiser Grad-CAM
        
        Args:
            model: Le modèle PyTorch
            target_layer_name (str): Nom de la couche cible
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.target_layer = self._get_target_layer()
        self.features = None
        self.gradients = None
        
        # Enregistrer les hooks
        self._register_hooks()
    
    def _get_target_layer(self):
        """Obtenir la couche cible"""
        layer = getattr(self.model.base_model, self.target_layer_name)
        return layer[-1] if isinstance(layer, (torch.nn.Sequential, torch.nn.ModuleList)) else layer
    
    def _register_hooks(self):
        """Enregistrer les hooks forward et backward"""
        def forward_hook(module, input, output):
            self.features = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_heatmap(self, image_tensor: torch.Tensor, 
                        class_idx: int) -> np.ndarray:
        """
        Générer une heatmap Grad-CAM
        
        Args:
            image_tensor (torch.Tensor): Image prétraitée
            class_idx (int): Index de la classe
            
        Returns:
            np.ndarray: Heatmap normalisée [0, 1]
        """
        # Forward pass
        self.model.eval()
        output = self.model(image_tensor)
        
        # Backward pass pour obtenir les gradients
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
        
        # Calculer la heatmap
        gradients = self.gradients[0]  # (channels, H, W)
        features = self.features[0]    # (channels, H, W)
        
        # Poids moyens
        weights = torch.mean(gradients, dim=[1, 2])
        
        # Heatmap pondérée
        heatmap = torch.zeros(features.shape[1:], device=features.device)
        for i in range(features.shape[0]):
            heatmap += weights[i] * features[i]
        
        # Normaliser
        heatmap = torch.clamp(heatmap, min=0)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        return heatmap.cpu().numpy()
    
    def overlay_heatmap(self, heatmap: np.ndarray, 
                       original_image: np.ndarray,
                       alpha: float = 0.5,
                       colormap: str = 'jet') -> np.ndarray:
        """
        Superposer la heatmap sur l'image originale
        
        Args:
            heatmap (np.ndarray): Heatmap Grad-CAM
            original_image (np.ndarray): Image originale (BGR)
            alpha (float): Transparence
            colormap (str): Colormap à utiliser
            
        Returns:
            np.ndarray: Image avec heatmap superposée
        """
        # Redimensionner la heatmap
        heatmap_resized = cv2.resize(heatmap, 
                                     (original_image.shape[1], original_image.shape[0]))
        
        # Appliquer la colormap
        heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8),
                                           cv2.COLORMAP_JET)
        
        # Superposer
        overlay = cv2.addWeighted(original_image, 1 - alpha, 
                                 heatmap_colored, alpha, 0)
        
        return overlay
    
    def explain(self, image_path: str, image_tensor: torch.Tensor,
               predicted_class_idx: int, alpha: float = 0.5) -> Dict:
        """
        Générer l'explication complète Grad-CAM
        
        Args:
            image_path (str): Chemin vers l'image
            image_tensor (torch.Tensor): Image prétraitée
            predicted_class_idx (int): Index de la classe prédite
            alpha (float): Transparence
            
        Returns:
            dict: {
                'heatmap': heatmap,
                'overlay': image avec heatmap,
                'original': image originale
            }
        """
        # Charger l'image originale
        original_image = ImageUtils.load_cv2_image(image_path)
        
        # Générer la heatmap
        heatmap = self.generate_heatmap(image_tensor, predicted_class_idx)
        
        # Créer l'overlay
        overlay = self.overlay_heatmap(heatmap, original_image, alpha=alpha)
        
        return {
            'heatmap': heatmap,
            'overlay': overlay,
            'original': original_image
        }


class LIMEExplainer:
    """
    Classe pour générer les explications LIME
    Explique les prédictions en utilisant une approximation locale
    """
    
    def __init__(self, model, device: torch.device):
        """
        Initialiser LIME
        
        Args:
            model: Le modèle PyTorch
            device (torch.device): Device (CPU ou CUDA)
        """
        self.model = model
        self.device = device
        self.explainer = lime_image.LimeImageExplainer()
    
    def _predict_function(self, images: np.ndarray) -> np.ndarray:
        """
        Fonction de prédiction pour LIME
        Convertit les images en tenseurs et retourne les prédictions
        
        Args:
            images (np.ndarray): Images normalisées [0, 1]
            
        Returns:
            np.ndarray: Probabilités
        """
        # Convertir en tenseurs
        tensors = []
        for img in images:
            tensor = torch.from_numpy(img).float()
            if tensor.dim() == 3 and tensor.shape[0] == 3:
                tensors.append(tensor.unsqueeze(0))
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).expand(3, -1, -1)
                tensors.append(tensor.unsqueeze(0))
        
        if not tensors:
            return np.zeros((len(images), 3))
        
        tensor_batch = torch.cat(tensors, dim=0).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(tensor_batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def explain(self, image_path: str, predicted_class_idx: int,
               num_samples: int = 1000, num_features: int = 10) -> Dict:
        """
        Générer l'explication LIME complète
        
        Args:
            image_path (str): Chemin vers l'image
            predicted_class_idx (int): Index de la classe prédite
            num_samples (int): Nombre d'échantillons LIME
            num_features (int): Nombre de features à visualiser
            
        Returns:
            dict: {
                'mask': mask image colorée,
                'top_label': top label,
                'overlay': Image avec overlay
            }
        """
        try:
            # Charger et normaliser l'image
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(original_image, (224, 224)).astype(np.float32) / 255.0
            
            # Générer l'explication LIME
            explanation = self.explainer.explain_instance(
                image_resized,
                self._predict_function,
                top_labels=1,
                hide_color=LIME_CONFIG['hide_color'],
                num_samples=num_samples
            )
            
            # Obtenir le top label
            top_label = explanation.top_labels[0]
            
            # Obtenir l'image avec le mask
            temp, mask = explanation.get_image_and_mask(
                top_label,
                positive_only=True,
                num_features=num_features,
                hide_rest=False
            )
            
            # Créer une image colorée à partir du mask
            # Le mask est binaire (0 ou 1), on va créer une heatmap
            mask_float = mask.astype(np.float32)
            
            # Redimensionner à 224x224 si nécessaire
            if mask_float.shape != (224, 224):
                mask_float = cv2.resize(mask_float, (224, 224))
            
            # Normaliser le mask
            mask_normalized = (mask_float / (mask_float.max() + 1e-8) * 255).astype(np.uint8)
            
            # Créer une heatmap colorée
            mask_colored = cv2.applyColorMap(mask_normalized, cv2.COLORMAP_JET)
            mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
            
            # Redimensionner à la taille originale
            original_h, original_w = original_image.shape[:2]
            mask_colored_resized = cv2.resize(mask_colored, (original_w, original_h))
            
            # Créer un overlay
            overlay = cv2.addWeighted(original_image, 0.6, mask_colored_resized, 0.4, 0)
            
            return {
                'mask': mask_colored_resized,
                'overlay': overlay,
                'top_label': top_label,
                'original_image': original_image
            }
        except Exception as e:
            print(f"[LIME Error] {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'mask': None,
                'overlay': None,
                'top_label': 0,
                'original_image': None
            }


class SHAPExplainer:
    """
    Classe pour générer les explications SHAP
    Explique les prédictions basées sur la théorie des jeux (valeurs Shapley)
    """
    
    def __init__(self, model, device: torch.device):
        """
        Initialiser SHAP
        
        Args:
            model: Le modèle PyTorch
            device (torch.device): Device (CPU ou CUDA)
        """
        self.model = model
        self.device = device
    
    def _predict_function(self, images: np.ndarray) -> np.ndarray:
        """
        Fonction de prédiction pour SHAP
        Convertit les images en tenseurs et retourne les prédictions
        
        Args:
            images (np.ndarray): Images normalisées [0, 1]
            
        Returns:
            np.ndarray: Probabilités pour la classe positive
        """
        # Convertir en tenseurs
        tensors = []
        for img in images:
            tensor = torch.from_numpy(img).float()
            if tensor.dim() == 3 and tensor.shape[0] == 3:
                tensors.append(tensor.unsqueeze(0))
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).expand(3, -1, -1)
                tensors.append(tensor.unsqueeze(0))
        
        if not tensors:
            return np.zeros((len(images),))
        
        tensor_batch = torch.cat(tensors, dim=0).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(tensor_batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Retourner les probabilités de la classe positive (index 1)
        return probabilities[:, 1].cpu().numpy()
    
    def explain(self, image_path: str, image_tensor: torch.Tensor,
               predicted_class_idx: int,
               background_samples: int = 50,
               num_samples: int = 100) -> Dict:
        """
        Générer l'explication SHAP basée sur les gradients
        
        Args:
            image_path (str): Chemin vers l'image
            image_tensor (torch.Tensor): Image prétraitée
            predicted_class_idx (int): Index de la classe prédite
            background_samples (int): Non utilisé (compatibilité)
            num_samples (int): Non utilisé (compatibilité)
            
        Returns:
            dict: {
                'overlay': Image avec heatmap d'importance,
                'original_image': Image originale,
                'feature_importance': Importance moyenne des features
            }
        """
        try:
            # Charger et normaliser l'image
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(original_image, (224, 224)).astype(np.float32) / 255.0
            
            # Convertir en tenseur et activer les gradients
            image_t = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).to(self.device)
            image_t.requires_grad = True
            
            # Forward pass
            with torch.enable_grad():
                outputs = self.model(image_t)
                if outputs.shape[1] > 1:
                    loss = outputs[0, predicted_class_idx]
                else:
                    loss = outputs[0, 0]
            
            # Backward pass pour obtenir les gradients
            if image_t.grad is not None:
                image_t.grad.zero_()
            
            loss.backward()
            
            # Récupérer les gradients
            gradients = image_t.grad.abs().cpu().numpy()[0]  # Shape: (3, 224, 224)
            
            # Moyenner sur les canaux
            feature_importance = np.mean(gradients, axis=0)  # Shape: (224, 224)
            
            # Normaliser
            feature_importance = (feature_importance - feature_importance.min()) / (feature_importance.max() - feature_importance.min() + 1e-8)
            
            # Créer un colormap
            feature_importance_colored = cv2.applyColorMap(
                (feature_importance * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            feature_importance_colored = cv2.cvtColor(feature_importance_colored, cv2.COLOR_BGR2RGB)
            
            # Redimensionner à la taille originale
            original_h, original_w = original_image.shape[:2]
            feature_importance_colored = cv2.resize(
                feature_importance_colored,
                (original_w, original_h)
            )
            
            # Overlay
            overlay = cv2.addWeighted(original_image, 0.6, feature_importance_colored, 0.4, 0)
            
            return {
                'feature_importance': feature_importance,
                'overlay': overlay,
                'original_image': original_image
            }
        except Exception as e:
            print(f"[SHAP Error] {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'feature_importance': None,
                'overlay': None,
                'original_image': None
            }


class XAIExplainer:
    """
    Classe principale pour les explications XAI
    Combine Grad-CAM, LIME et SHAP
    """
    
    def __init__(self, model, device: torch.device):
        """
        Initialiser l'explaineur XAI
        
        Args:
            model: Le modèle PyTorch
            device (torch.device): Device (CPU ou CUDA)
        """
        self.model = model
        self.device = device
        self.gradcam = GradCAMExplainer(model)
        self.lime = LIMEExplainer(model, device)
        self.shap = SHAPExplainer(model, device)
    
    def explain_complete(self, image_path: str, image_tensor: torch.Tensor,
                        predicted_class_idx: int,
                        alpha: float = 0.5,
                        num_samples: int = 1000,
                        include_shap: bool = True,
                        shap_background_samples: int = 50) -> Dict:
        """
        Générer les explications complètes (Grad-CAM + LIME + optionnellement SHAP)
        
        Args:
            image_path (str): Chemin vers l'image
            image_tensor (torch.Tensor): Image prétraitée
            predicted_class_idx (int): Index de la classe prédite
            alpha (float): Transparence pour Grad-CAM
            num_samples (int): Nombre d'échantillons LIME
            include_shap (bool): Inclure SHAP dans les explications (peut être lent)
            shap_background_samples (int): Nombre d'échantillons de fond pour SHAP
            
        Returns:
            dict: Explications Grad-CAM, LIME et optionnellement SHAP
        """
        results = {
            'gradcam': self.gradcam.explain(image_path, image_tensor,
                                           predicted_class_idx, alpha=alpha),
            'lime': self.lime.explain(image_path, predicted_class_idx,
                                     num_samples=num_samples)
        }
        
        if include_shap:
            results['shap'] = self.shap.explain(
                image_path, 
                image_tensor,
                predicted_class_idx,
                background_samples=shap_background_samples,
                num_samples=num_samples
            )
        
        return results
    
    def get_supported_methods(self):
        """Obtenir les méthodes XAI supportées"""
        return ['grad_cam', 'lime', 'shap']

