import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
from .audio_models import build_mobilenet_model, build_vgg16_model, build_resnet_model
from .spectrogram_converter import create_spectrogram


class DeepfakeAudioDetector:
    """
    Classe pour la détection de deepfake audio
    Supporte MobileNet, VGG16, ResNet50
    """
    
    def __init__(self, model_type='mobilenet', weights_path=None):
        """
        Initialiser le détecteur avec un type de modèle
        
        Args:
            model_type (str): 'mobilenet', 'vgg16', ou 'resnet50'
            weights_path (str): Chemin vers les poids entraînés (optionnel)
        """
        self.model_type = model_type
        self.model = self._load_model()
        
        if weights_path and os.path.exists(weights_path):
            self.model.load_weights(weights_path)
    
    def _load_model(self):
        """Charger le modèle architectural basé sur le type spécifié"""
        if self.model_type == 'mobilenet':
            return build_mobilenet_model()
        elif self.model_type == 'vgg16':
            return build_vgg16_model()
        elif self.model_type == 'resnet50':
            return build_resnet_model()
        else:
            raise ValueError(f"Modèle non reconnu: {self.model_type}")
    
    def preprocess_audio(self, audio_path, temp_dir='/tmp'):
        """
        Prétraiter un fichier audio
        
        Args:
            audio_path (str): Chemin vers le fichier .wav
            temp_dir (str): Répertoire temporaire pour stocker le spectrogramme
            
        Returns:
            np.ndarray: Image normalisée prête pour le modèle
        """
        # Créer le répertoire temporaire s'il n'existe pas
        os.makedirs(temp_dir, exist_ok=True)
        
        # Convertir audio en spectrogramme
        spec_path = os.path.join(temp_dir, 'temp_spectrogram.png')
        create_spectrogram(audio_path, spec_path)
        
        # Charger et redimensionner l'image
        img = Image.open(spec_path).convert('RGB')
        img = img.resize((224, 224))
        
        # Normaliser
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Nettoyer
        if os.path.exists(spec_path):
            os.remove(spec_path)
        
        return img_array, spec_path
    
    def predict(self, audio_path):
        """
        Faire une prédiction sur un fichier audio
        
        Args:
            audio_path (str): Chemin vers le fichier .wav
            
        Returns:
            dict: Contient les probabilités et le label prédit
        """
        img_array, spec_path = self.preprocess_audio(audio_path)
        
        # Prédiction
        prediction = self.model.predict(img_array, verbose=0)
        
        # Extraire les probabilités (supposant sortie softmax pour 2 classes)
        if prediction.shape[1] == 2:
            real_prob = float(prediction[0][0])
            fake_prob = float(prediction[0][1])
        else:
            # Si une seule sortie (sigmoid)
            real_prob = 1 - float(prediction[0][0])
            fake_prob = float(prediction[0][0])
        
        predicted_label = 'REAL' if real_prob > fake_prob else 'FAKE'
        confidence = max(real_prob, fake_prob)
        
        return {
            'real_probability': real_prob,
            'fake_probability': fake_prob,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'model_type': self.model_type
        }
    
    def batch_predict(self, audio_paths):
        """
        Faire des prédictions sur plusieurs fichiers audio
        
        Args:
            audio_paths (list): Liste des chemins vers les fichiers .wav
            
        Returns:
            list: Liste des résultats de prédiction
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                result['audio_path'] = audio_path
                results.append(result)
            except Exception as e:
                results.append({
                    'audio_path': audio_path,
                    'error': str(e)
                })
        return results
    
    def get_model_info(self):
        """Obtenir les informations du modèle"""
        return {
            'type': self.model_type,
            'total_params': self.model.count_params(),
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape
        }
