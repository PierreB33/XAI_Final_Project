"""
Lung Cancer Detection Backend - Model & Prediction
"""

import os
import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Tuple
import warnings

from .config import MODEL_CONFIG, PREPROCESSING_CONFIG, DEVICE_CONFIG
from .utils import ImagePreprocessor, ImageUtils, DeviceUtils

warnings.filterwarnings('ignore')


class LungCancerResNet50(nn.Module):
    """
    ResNet50 pré-entraîné pour la détection du cancer du poumon
    Architecture: ResNet50 + couches fully connected personnalisées
    Classes: Adenocarcinoma, Benign, Squamous Cell Carcinoma
    """
    
    def __init__(self, num_classes: int = 3):
        """
        Initialiser le modèle
        
        Args:
            num_classes (int): Nombre de classes (3)
        """
        super(LungCancerResNet50, self).__init__()
        
        # ResNet50 pré-entraîné
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remplacer la couche finale
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.base_model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Obtenir les features avant la classification finale"""
        return self.base_model.avgpool(self.base_model.layer4(
            self.base_model.layer3(self.base_model.layer2(
                self.base_model.layer1(self.base_model.relu(
                    self.base_model.bn1(self.base_model.conv1(x))
                ))
            ))
        )).flatten(1)


class Predictor:
    """
    Classe principale pour la prédiction du cancer du poumon
    Gère le chargement du modèle et la prédiction sur des images
    """
    
    def __init__(self, model_path: str, device: torch.device = None, 
                 use_default_weights: bool = False):
        """
        Initialiser le prédicteur
        
        Args:
            model_path (str): Chemin vers les poids du modèle (.pth)
            device (torch.device): Device (CPU ou CUDA)
            use_default_weights (bool): Utiliser les poids ImageNet par défaut
        """
        self.device = device or DeviceUtils.get_device(DEVICE_CONFIG['use_cuda'])
        self.model_path = model_path
        self.model = None
        self.preprocessor = ImagePreprocessor(
            image_size=PREPROCESSING_CONFIG['image_size'],
            mean=PREPROCESSING_CONFIG['mean'],
            std=PREPROCESSING_CONFIG['std']
        )
        self.class_names = MODEL_CONFIG['class_names']
        self.class_indices = MODEL_CONFIG['class_indices']
        
        # Charger le modèle
        self._load_model(use_default_weights)
    
    def _load_model(self, use_default_weights: bool = False) -> None:
        """
        Charger le modèle
        
        Args:
            use_default_weights (bool): Utiliser les poids par défaut
        """
        self.model = LungCancerResNet50(num_classes=MODEL_CONFIG['num_classes'])
        
        if not use_default_weights and os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ Modèle chargé depuis: {self.model_path}")
        else:
            print(f"⚠️  Poids pré-entraînés utilisés (ImageNet)")
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image_path: str) -> Dict:
        """
        Prédire la classe d'une image
        
        Args:
            image_path (str): Chemin vers l'image
            
        Returns:
            dict: {
                'predicted_class': str,
                'predicted_class_idx': int,
                'probabilities': dict,
                'confidence': float,
                'all_classes': dict
            }
        """
        # Prétraiter l'image
        image_tensor = self.preprocessor.preprocess(image_path, self.device)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
        
        # Préparer les résultats
        prob_dict = {
            self.class_names[i]: probabilities[i].item()
            for i in range(len(self.class_names))
        }
        
        return {
            'predicted_class': self.class_names[predicted_idx],
            'predicted_class_idx': predicted_idx,
            'probabilities': prob_dict,
            'confidence': confidence,
            'all_classes': self.class_names
        }
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Prédire sur un batch d'images
        
        Args:
            image_paths (list): Liste des chemins vers les images
            
        Returns:
            list: Liste des résultats
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        return results
    
    def get_model_info(self) -> Dict:
        """Obtenir les informations du modèle"""
        return {
            'model_type': 'ResNet50',
            'num_classes': MODEL_CONFIG['num_classes'],
            'class_names': self.class_names,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'device': str(self.device),
            'input_size': PREPROCESSING_CONFIG['image_size']
        }
    
    def get_layer(self, layer_name: str) -> nn.Module:
        """Obtenir une couche du modèle"""
        return getattr(self.model.base_model, layer_name)
    
    def register_hook(self, layer_name: str, hook_fn) -> None:
        """
        Enregistrer un hook sur une couche
        
        Args:
            layer_name (str): Nom de la couche
            hook_fn: Fonction du hook
        """
        layer = self.get_layer(layer_name)
        layer.register_forward_hook(hook_fn)
