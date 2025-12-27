import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
from lime.lime_image import LimeImageExplainer
from lime import lime_image
import shap
from .spectrogram_converter import create_spectrogram


class XAIExplainer:
    """
    Classe pour implémenter les techniques XAI:
    - Grad-CAM
    - LIME
    - SHAP
    """
    
    def __init__(self, model, model_type='mobilenet'):
        """
        Initialiser l'explaineur XAI
        
        Args:
            model: Le modèle TensorFlow/Keras à expliquer
            model_type (str): Type du modèle ('mobilenet', 'vgg16', 'resnet50')
        """
        self.model = model
        self.model_type = model_type
        self.grad_model = None
    
    def _load_spectrogram(self, audio_path, temp_dir='/tmp'):
        """Charger un spectrogramme depuis un fichier audio"""
        os.makedirs(temp_dir, exist_ok=True)
        spec_path = os.path.join(temp_dir, 'temp_spectrogram.png')
        create_spectrogram(audio_path, spec_path)
        
        img = Image.open(spec_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        
        if os.path.exists(spec_path):
            os.remove(spec_path)
        
        return img_array
    
    # ========== GRAD-CAM ==========
    def grad_cam(self, audio_path, class_index=1, layer_name=None):
        """
        Implémenter Grad-CAM pour la visualisation des régions activées
        
        Args:
            audio_path (str): Chemin vers le fichier .wav
            class_index (int): Index de la classe pour laquelle générer la heatmap
            layer_name (str): Nom de la couche de convololution (auto-détection si None)
            
        Returns:
            dict: Contient la heatmap et l'image superposée
        """
        img_array = self._load_spectrogram(audio_path)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Auto-détection de la couche si non spécifiée
        if layer_name is None:
            layer_name = self._get_last_conv_layer()
        
        # Créer le modèle Grad-CAM
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(layer_name).output, self.model.output]
        )
        
        # Enregistrer la bande pour les gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]
        
        # Calculer les gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Calculer les poids moyens
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Générer la heatmap
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(pooled_grads[..., None] * conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)
        heatmap = cv2.resize(heatmap, (224, 224))
        
        # Superposer sur l'image originale
        original_img = (img_array[0] * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        superposed = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
        
        return {
            'method': 'Grad-CAM',
            'heatmap': heatmap,
            'superposed_image': superposed,
            'layer': layer_name,
            'class_index': class_index
        }
    
    def _get_last_conv_layer(self):
        """Trouver le nom de la dernière couche de convolution"""
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        return None
    
    # ========== LIME ==========
    def lime_explanation(self, audio_path, num_samples=1000):
        """
        Implémenter LIME pour les explications locales
        
        Args:
            audio_path (str): Chemin vers le fichier .wav
            num_samples (int): Nombre d'échantillons pour LIME
            
        Returns:
            dict: Contient l'explication LIME
        """
        img_array = self._load_spectrogram(audio_path)
        
        # Créer l'explaineur LIME
        explainer = LimeImageExplainer()
        
        # Fonction de prédiction pour LIME
        def predict_fn(images):
            # Normaliser les images
            images = images / 255.0 if images.max() > 1 else images
            return self.model.predict(images, verbose=0)
        
        # Générer l'explication
        explanation = explainer.explain_instance(
            img_array,
            predict_fn,
            top_labels=2,
            num_samples=num_samples,
            random_seed=42
        )
        
        # Obtenir l'image avec les régions importantes
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        
        return {
            'method': 'LIME',
            'explanation': explanation,
            'highlighted_image': temp,
            'mask': mask,
            'num_samples': num_samples
        }
    
    # ========== SHAP ==========
    def shap_explanation(self, audio_path, background_samples=100):
        """
        Implémenter SHAP pour les explications basées sur la théorie des jeux
        
        Args:
            audio_path (str): Chemin vers le fichier .wav
            background_samples (int): Nombre d'échantillons de fond
            
        Returns:
            dict: Contient l'explication SHAP
        """
        img_array = self._load_spectrogram(audio_path)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Créer des données de fond (images bruitées)
        background = np.random.rand(background_samples, 224, 224, 3)
        
        # Fonction de prédiction pour SHAP
        def predict_fn(images):
            images = images / 255.0 if images.max() > 1 else images
            preds = self.model.predict(images, verbose=0)
            # Retourner les probabilités de la classe "fake"
            if preds.shape[1] == 2:
                return preds[:, 1]
            else:
                return preds[:, 0]
        
        # Créer l'explaineur SHAP avec un sous-ensemble
        sample_indices = np.random.choice(background.shape[0], 50, replace=False)
        explainer = shap.KernelExplainer(predict_fn, background[sample_indices])
        
        # Générer les valeurs SHAP
        shap_values = explainer.shap_values(img_array)
        
        return {
            'method': 'SHAP',
            'shap_values': shap_values,
            'base_value': explainer.expected_value,
            'num_features': shap_values.shape[-1]
        }
    
    # ========== COMPARAISON ==========
    def compare_methods(self, audio_path):
        """
        Générer des explications avec toutes les méthodes
        
        Args:
            audio_path (str): Chemin vers le fichier .wav
            
        Returns:
            dict: Contient toutes les explications
        """
        results = {
            'audio_path': audio_path,
            'methods': {}
        }
        
        try:
            grad_cam_result = self.grad_cam(audio_path)
            results['methods']['grad_cam'] = grad_cam_result
        except Exception as e:
            results['methods']['grad_cam'] = {'error': str(e)}
        
        try:
            lime_result = self.lime_explanation(audio_path)
            results['methods']['lime'] = lime_result
        except Exception as e:
            results['methods']['lime'] = {'error': str(e)}
        
        try:
            shap_result = self.shap_explanation(audio_path)
            results['methods']['shap'] = shap_result
        except Exception as e:
            results['methods']['shap'] = {'error': str(e)}
        
        return results
    
    def get_supported_methods(self):
        """Obtenir les méthodes XAI supportées"""
        return ['grad_cam', 'lime', 'shap']
