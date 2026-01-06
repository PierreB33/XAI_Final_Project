"""
Lung Cancer Detection Backend - Utilities
"""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple, Union
import matplotlib.pyplot as plt


class ImagePreprocessor:
    """Classe pour prétraiter les images"""
    
    def __init__(self, image_size: int = 224, 
                 mean: list = None, std: list = None):
        """
        Initialiser le préprocesseur d'images
        
        Args:
            image_size (int): Taille cible (224x224)
            mean (list): Moyennes de normalisation ImageNet
            std (list): Écarts-types de normalisation ImageNet
        """
        self.image_size = image_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def preprocess(self, image_path: str, device: torch.device) -> torch.Tensor:
        """
        Prétraiter une image
        
        Args:
            image_path (str): Chemin vers l'image
            device (torch.device): Device (CPU/GPU)
            
        Returns:
            torch.Tensor: Image prétraitée (batch)
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(device)
    
    def preprocess_from_cv2(self, cv2_image: np.ndarray, 
                            device: torch.device) -> torch.Tensor:
        """
        Prétraiter une image depuis OpenCV
        
        Args:
            cv2_image (np.ndarray): Image OpenCV (BGR)
            device (torch.device): Device (CPU/GPU)
            
        Returns:
            torch.Tensor: Image prétraitée
        """
        # Convertir BGR → RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        image_tensor = self.transform(pil_image).unsqueeze(0)
        return image_tensor.to(device)
    
    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Dénormaliser une image
        
        Args:
            tensor (torch.Tensor): Image normalisée
            
        Returns:
            np.ndarray: Image dénormalisée (0-255)
        """
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        
        denorm = tensor[0] * std + mean
        denorm = torch.clamp(denorm, 0, 1)
        return (denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


class ImageUtils:
    """Utilitaires pour les images"""
    
    @staticmethod
    def load_cv2_image(image_path: str) -> np.ndarray:
        """Charger une image avec OpenCV"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        return image
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> None:
        """Sauvegarder une image"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
    
    @staticmethod
    def overlay_text(image: np.ndarray, text: str, 
                    position: Tuple[int, int] = (10, 30),
                    font_scale: float = 1.0,
                    font_color: Tuple[int, int, int] = (0, 255, 0),
                    thickness: int = 2) -> np.ndarray:
        """
        Ajouter du texte sur une image
        
        Args:
            image (np.ndarray): Image
            text (str): Texte à ajouter
            position (Tuple): Position (x, y)
            font_scale (float): Échelle de police
            font_color (Tuple): Couleur BGR
            thickness (int): Épaisseur du texte
            
        Returns:
            np.ndarray: Image avec texte
        """
        image_copy = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_copy, text, position, font, font_scale, 
                   font_color, thickness, cv2.LINE_AA)
        return image_copy
    
    @staticmethod
    def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Redimensionner une image"""
        return cv2.resize(image, size)
    
    @staticmethod
    def get_image_info(image_path: str) -> dict:
        """Obtenir les informations d'une image"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        return {
            'shape': image.shape,
            'height': image.shape[0],
            'width': image.shape[1],
            'channels': image.shape[2],
            'dtype': str(image.dtype),
            'file_size': os.path.getsize(image_path)
        }


class FileUtils:
    """Utilitaires pour les fichiers"""
    
    @staticmethod
    def ensure_directory(directory: str) -> str:
        """Créer un répertoire s'il n'existe pas"""
        os.makedirs(directory, exist_ok=True)
        return directory
    
    @staticmethod
    def get_file_name(file_path: str) -> str:
        """Obtenir le nom du fichier sans extension"""
        return os.path.splitext(os.path.basename(file_path))[0]
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """Obtenir l'extension du fichier"""
        return os.path.splitext(file_path)[1]
    
    @staticmethod
    def join_path(*args) -> str:
        """Joindre des chemins"""
        return os.path.join(*args)


class VisualizationUtils:
    """Utilitaires pour la visualisation"""
    
    @staticmethod
    def create_bar_chart(labels: list, values: list, 
                        title: str = "Feature Contributions",
                        output_path: str = None) -> None:
        """
        Créer un graphique en barres
        
        Args:
            labels (list): Labels des barres
            values (list): Valeurs des barres
            title (str): Titre du graphique
            output_path (str): Chemin de sauvegarde (optionnel)
        """
        plt.figure(figsize=(10, 6))
        plt.barh(labels, values, color='steelblue')
        plt.xlabel('Contribution')
        plt.ylabel('Features')
        plt.title(title)
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
        
        plt.close()
    
    @staticmethod
    def create_comparison_grid(images: dict, 
                              titles: dict = None,
                              output_path: str = None) -> None:
        """
        Créer une grille de comparaison d'images
        
        Args:
            images (dict): Dictionnaire {clé: image_array}
            titles (dict): Dictionnaire {clé: titre}
            output_path (str): Chemin de sauvegarde
        """
        n_images = len(images)
        fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
        
        if n_images == 1:
            axes = [axes]
        
        for idx, (key, image) in enumerate(images.items()):
            ax = axes[idx]
            if image.ndim == 3 and image.shape[2] == 3:
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(image, cmap='gray')
            
            title = titles.get(key, key) if titles else key
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
        
        plt.close()


class DeviceUtils:
    """Utilitaires pour le device (CPU/GPU)"""
    
    @staticmethod
    def get_device(use_cuda: bool = True) -> torch.device:
        """Obtenir le device (CPU ou CUDA)"""
        if use_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    @staticmethod
    def get_device_info() -> dict:
        """Obtenir les informations du device"""
        return {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'pytorch_version': torch.__version__
        }
