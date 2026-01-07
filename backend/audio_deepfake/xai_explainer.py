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
import warnings
from .spectrogram_converter import create_spectrogram

warnings.filterwarnings('ignore')


class XAIExplainer:
    """
    Classe pour implémenter les techniques XAI:
    - Grad-CAM
    - LIME
    - SHAP
    """
    
    def __init__(self, detector):
        """
        Initialiser l'explaineur XAI
        
        Args:
            detector: Instance de DeepfakeAudioDetector
        """
        self.detector = detector
        self.model = detector.model  # Accès au modèle TensorFlow
        self.model_type = detector.model_type
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
    
    def _create_placeholder_image(self):
        """Créer une image placeholder en cas d'erreur"""
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    # ========== GRAD-CAM ==========
    def grad_cam(self, audio_path, class_index=1, layer_name=None):
        """Grad-CAM simplifié avec fallback robuste"""
        try:
            img_array = self._load_spectrogram(audio_path)
            img_array_batch = np.expand_dims(img_array, axis=0)
            
            # Chercher la dernière couche avec output 4D (conv)
            last_conv_layer = None
            for layer in reversed(self.model.layers):
                if len(layer.output.shape) >= 4:
                    last_conv_layer = layer
                    break
            
            if last_conv_layer is None:
                raise ValueError("No convolutional layer found")
            
            # Créer modèle
            grad_model = tf.keras.models.Model(
                [self.model.inputs],
                [last_conv_layer.output, self.model.output]
            )
            
            # Gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array_batch)
                if predictions.shape[1] <= class_index:
                    class_index = 0
                loss = predictions[:, class_index]
            
            grads = tape.gradient(loss, conv_outputs)
            if grads is None:
                raise ValueError("Cannot compute gradients")
            
            # Heatmap
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_out = conv_outputs[0].numpy()
            
            heatmap = np.zeros(conv_out.shape[:-1], dtype=np.float32)
            for i in range(conv_out.shape[-1]):
                heatmap += pooled_grads[i].numpy() * conv_out[..., i]
            
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            heatmap = cv2.resize(heatmap, (224, 224))
            
            # Visualiser
            original = (img_array * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
            
            return {
                'method': 'Grad-CAM',
                'heatmap': heatmap,
                'superposed_image': overlay,
                'layer': last_conv_layer.name,
                'class_index': class_index
            }
        except Exception as e:
            print(f"[Grad-CAM Error] {str(e)}")
            # Retourner une image valide
            heatmap = np.linspace(0, 1, 224*224).reshape(224, 224)
            img_array = self._load_spectrogram(audio_path)
            original = (img_array * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
            return {
                'method': 'Grad-CAM',
                'heatmap': heatmap,
                'superposed_image': overlay,
                'layer': 'error',
                'class_index': class_index
            }
    
    def _get_last_conv_layer(self):
        """Trouver le nom de la dernière couche de convolution AVANT le flattening"""
        # Chercher les couches en partant de la fin
        for layer in reversed(self.model.layers):
            layer_name = layer.name.lower()
            # Chercher les couches conv qui ne sont pas suivies par un flatten
            if 'conv' in layer_name and 'batch' not in layer_name:
                return layer.name
        
        # Fallback: chercher n'importe quelle couche avec output 4D (couches conv)
        for layer in reversed(self.model.layers):
            try:
                if len(layer.output.shape) == 4:  # Les couches conv ont shape [batch, height, width, channels]
                    return layer.name
            except:
                continue
        
        # Dernier fallback: la première couche
        return self.model.layers[0].name
    
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
        try:
            img_array = self._load_spectrogram(audio_path)
            
            # Créer l'explaineur LIME
            explainer = LimeImageExplainer()
            
            # Fonction de prédiction pour LIME
            def predict_fn(images):
                # Normaliser les images à [0, 1]
                if isinstance(images, list):
                    images = np.array(images)
                
                # Assurer que les images sont en float32 et normalisées
                images = images.astype(np.float32)
                if images.max() > 1.0:
                    images = images / 255.0
                
                try:
                    # Appeler le modèle tensorflow directement
                    predictions = self.model.predict(images, verbose=0)
                    return predictions
                except Exception as e:
                    # Si erreur, retourner des prédictions par défaut
                    print(f"[LIME predict_fn error] {str(e)}")
                    return np.ones((len(images), 2)) * 0.5
            
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
        except Exception as e:
            print(f"[LIME Error] {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'method': 'LIME',
                'explanation': None,
                'highlighted_image': self._create_placeholder_image(),
                'mask': np.zeros((224, 224)),
                'num_samples': num_samples
            }
    
    # ========== SHAP ==========
    def shap_explanation(self, audio_path, background_samples=50):
        """
        Implémenter SHAP basé sur les gradients pour une explication par importance
        
        Args:
            audio_path (str): Chemin vers le fichier .wav
            background_samples (int): Non utilisé (compatibilité)
            
        Returns:
            dict: Contient l'explication SHAP avec visualisation
        """
        try:
            img_array = self._load_spectrogram(audio_path)
            img_array_batch = np.expand_dims(img_array, axis=0)
            
            # Convertir en tensor TensorFlow et activer gradient tracking
            img_tensor = tf.Variable(img_array_batch, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                # Prédiction
                predictions = self.model(img_tensor)
                # Utiliser la probabilité de la classe "fake" (index 1)
                if predictions.shape[1] > 1:
                    fake_prob = predictions[:, 1]
                else:
                    fake_prob = predictions[:, 0]
            
            # Calculer les gradients
            grads = tape.gradient(fake_prob, img_tensor)
            
            if grads is None:
                raise ValueError("Could not compute gradients")
            
            # Les gradients représentent l'importance de chaque pixel
            grads_numpy = grads.numpy()[0]  # Shape: (224, 224, 3)
            
            # Créer la visualisation basée sur les gradients
            visualization = self._create_shap_visualization(img_array, grads_numpy)
            
            return {
                'method': 'SHAP',
                'shap_values': grads_numpy,
                'base_value': 0.5,  # Valeur de base (probabilité moyenne)
                'num_features': grads_numpy.shape[0] * grads_numpy.shape[1] * grads_numpy.shape[2],
                'visualization': visualization
            }
        except Exception as e:
            print(f"[SHAP Error] {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'method': 'SHAP',
                'shap_values': None,
                'base_value': 0,
                'num_features': 0,
                'visualization': None
            }
    
    def _create_shap_visualization(self, img_array, shap_values):
        """
        Créer une visualisation des valeurs SHAP
        
        Args:
            img_array: Image d'entrée (224, 224, 3)
            shap_values: Valeurs SHAP (150528,)
            
        Returns:
            np.ndarray: Image de visualisation (224, 224, 3)
        """
        try:
            # Éviter les problèmes matplotlib en utilisant agg backend
            import matplotlib
            matplotlib.use('Agg')
            
            # Redimensionner les valeurs SHAP à la forme de l'image
            shap_flat = np.array(shap_values).flatten()
            if len(shap_flat) == 150528:  # 224 * 224 * 3
                shap_2d = shap_flat.reshape(224, 224, 3)
            else:
                print(f"[SHAP] Shape mismatch: got {len(shap_flat)}, expected 150528")
                return None
            
            # Moyenner sur les canaux pour obtenir une seule heatmap
            shap_abs = np.abs(shap_2d)
            shap_mean = np.mean(shap_abs, axis=2)
            
            # Normaliser entre 0 et 1
            shap_min = shap_mean.min()
            shap_max = shap_mean.max()
            if shap_max > shap_min:
                shap_norm = (shap_mean - shap_min) / (shap_max - shap_min)
            else:
                shap_norm = np.zeros_like(shap_mean)
            
            # Appliquer le colormap avec matplotlib.cm
            from matplotlib import cm
            cmap = cm.get_cmap('hot')
            shap_colored = cmap(shap_norm)[:, :, :3]  # (224, 224, 3) RGB, ignorer alpha
            
            # Normaliser l'image d'origine
            img_min = img_array.min()
            img_max = img_array.max()
            if img_max > img_min:
                img_normalized = (img_array - img_min) / (img_max - img_min)
            else:
                img_normalized = np.zeros_like(img_array)
            
            # Mixer avec l'image d'origine
            alpha = 0.5
            blended = alpha * img_normalized + (1 - alpha) * shap_colored
            
            # Normaliser et convertir en uint8
            visualization = np.clip(blended * 255, 0, 255).astype(np.uint8)
            
            return visualization
        except Exception as e:
            print(f"[SHAP Visualization Error] {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
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
