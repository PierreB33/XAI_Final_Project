"""
Validation et gestion des erreurs pour le système de détection
"""

import os
import librosa
from pathlib import Path


class AudioValidator:
    """Validateur pour les fichiers audio"""
    
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.ogg', '.flac']
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    MIN_DURATION = 0.1  # 100ms
    MAX_DURATION = 600  # 10 minutes
    
    @staticmethod
    def validate_file(file_path):
        """
        Valider un fichier audio
        
        Args:
            file_path (str): Chemin vers le fichier
            
        Returns:
            tuple: (is_valid, error_message)
        """
        # Vérifier l'existence
        if not os.path.exists(file_path):
            return False, f"Fichier non trouvé: {file_path}"
        
        # Vérifier l'extension
        ext = Path(file_path).suffix.lower()
        if ext not in AudioValidator.SUPPORTED_FORMATS:
            return False, f"Format non supporté: {ext}. Supportés: {AudioValidator.SUPPORTED_FORMATS}"
        
        # Vérifier la taille
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "Fichier vide"
        
        if file_size > AudioValidator.MAX_FILE_SIZE:
            return False, f"Fichier trop volumineux: {file_size / 1024 / 1024:.2f} MB (max: {AudioValidator.MAX_FILE_SIZE / 1024 / 1024:.2f} MB)"
        
        # Vérifier que c'est un audio valide
        try:
            y, sr = librosa.load(file_path, sr=None, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
            
            if duration < AudioValidator.MIN_DURATION:
                return False, f"Audio trop court: {duration:.2f}s (min: {AudioValidator.MIN_DURATION}s)"
            
            if duration > AudioValidator.MAX_DURATION:
                return False, f"Audio trop long: {duration:.2f}s (max: {AudioValidator.MAX_DURATION}s)"
            
            return True, None
            
        except Exception as e:
            return False, f"Erreur lors de la lecture audio: {str(e)}"
    
    @staticmethod
    def get_audio_metadata(file_path):
        """
        Obtenir les métadonnées audio
        
        Args:
            file_path (str): Chemin vers le fichier
            
        Returns:
            dict: Métadonnées ou None en cas d'erreur
        """
        try:
            y, sr = librosa.load(file_path, sr=None, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'num_samples': len(y),
                'file_size': os.path.getsize(file_path),
                'is_mono': True,
                'dtype': str(y.dtype)
            }
        except Exception as e:
            return None


class ModelValidator:
    """Validateur pour les modèles"""
    
    VALID_MODELS = ['mobilenet', 'vgg16', 'resnet50']
    VALID_XAI_METHODS = ['grad_cam', 'lime', 'shap']
    
    @staticmethod
    def validate_model_type(model_type):
        """
        Valider le type de modèle
        
        Args:
            model_type (str): Type de modèle
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not isinstance(model_type, str):
            return False, f"Type de modèle doit être une string, reçu: {type(model_type)}"
        
        if model_type.lower() not in ModelValidator.VALID_MODELS:
            return False, f"Modèle '{model_type}' non reconnu. Valides: {ModelValidator.VALID_MODELS}"
        
        return True, None
    
    @staticmethod
    def validate_xai_method(method):
        """
        Valider la méthode XAI
        
        Args:
            method (str): Méthode XAI
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not isinstance(method, str):
            return False, f"Méthode XAI doit être une string, reçu: {type(method)}"
        
        if method.lower() not in ModelValidator.VALID_XAI_METHODS:
            return False, f"Méthode '{method}' non reconnue. Valides: {ModelValidator.VALID_XAI_METHODS}"
        
        return True, None
    
    @staticmethod
    def validate_weights_file(weights_path):
        """
        Valider un fichier de poids
        
        Args:
            weights_path (str): Chemin vers les poids
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if weights_path is None:
            return True, None
        
        if not os.path.exists(weights_path):
            return False, f"Fichier de poids non trouvé: {weights_path}"
        
        if not weights_path.endswith(('.h5', '.pt', '.ckpt')):
            return False, f"Format de poids non reconnu: {Path(weights_path).suffix}"
        
        return True, None


class InputValidator:
    """Validateur pour les entrées utilisateur"""
    
    @staticmethod
    def validate_prediction_input(audio_path, model_type, weights_path=None):
        """
        Valider les entrées pour une prédiction
        
        Args:
            audio_path (str): Chemin vers l'audio
            model_type (str): Type de modèle
            weights_path (str): Chemin vers les poids
            
        Returns:
            tuple: (is_valid, error_message)
        """
        # Valider l'audio
        is_valid, error = AudioValidator.validate_file(audio_path)
        if not is_valid:
            return False, error
        
        # Valider le modèle
        is_valid, error = ModelValidator.validate_model_type(model_type)
        if not is_valid:
            return False, error
        
        # Valider les poids
        is_valid, error = ModelValidator.validate_weights_file(weights_path)
        if not is_valid:
            return False, error
        
        return True, None
    
    @staticmethod
    def validate_xai_input(audio_path, method):
        """
        Valider les entrées pour une explication XAI
        
        Args:
            audio_path (str): Chemin vers l'audio
            method (str): Méthode XAI
            
        Returns:
            tuple: (is_valid, error_message)
        """
        # Valider l'audio
        is_valid, error = AudioValidator.validate_file(audio_path)
        if not is_valid:
            return False, error
        
        # Valider la méthode
        is_valid, error = ModelValidator.validate_xai_method(method)
        if not is_valid:
            return False, error
        
        return True, None


class ErrorHandler:
    """Gestion centralisée des erreurs"""
    
    @staticmethod
    def handle_audio_error(error, audio_path):
        """
        Gérer une erreur audio
        
        Args:
            error (Exception): L'exception
            audio_path (str): Chemin du fichier
            
        Returns:
            dict: Rapport d'erreur
        """
        return {
            'success': False,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'audio_path': audio_path,
            'suggestion': 'Vérifiez que le fichier est un audio valide au format .wav, .mp3, .ogg ou .flac'
        }
    
    @staticmethod
    def handle_model_error(error, model_type):
        """
        Gérer une erreur de modèle
        
        Args:
            error (Exception): L'exception
            model_type (str): Type de modèle
            
        Returns:
            dict: Rapport d'erreur
        """
        return {
            'success': False,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'model_type': model_type,
            'suggestion': 'Vérifiez que les dépendances sont installées et que le modèle est disponible'
        }
    
    @staticmethod
    def handle_xai_error(error, method, audio_path):
        """
        Gérer une erreur XAI
        
        Args:
            error (Exception): L'exception
            method (str): Méthode XAI
            audio_path (str): Chemin du fichier
            
        Returns:
            dict: Rapport d'erreur
        """
        return {
            'success': False,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'method': method,
            'audio_path': audio_path,
            'suggestion': f'La méthode {method} a échoué. Essayez une autre méthode XAI ou vérifiez l\'audio'
        }


class SafeDeepfakeDetector:
    """Wrapper sécurisé du détecteur avec validation"""
    
    def __init__(self, model_type='mobilenet', weights_path=None):
        """Initialiser avec validation"""
        # Valider les entrées
        is_valid, error = ModelValidator.validate_model_type(model_type)
        if not is_valid:
            raise ValueError(error)
        
        is_valid, error = ModelValidator.validate_weights_file(weights_path)
        if not is_valid:
            raise ValueError(error)
        
        from deepfake_detector import DeepfakeAudioDetector
        self.detector = DeepfakeAudioDetector(model_type, weights_path)
    
    def predict_safe(self, audio_path):
        """
        Faire une prédiction avec gestion d'erreur
        
        Args:
            audio_path (str): Chemin vers l'audio
            
        Returns:
            dict: Résultat ou rapport d'erreur
        """
        # Valider l'entrée
        is_valid, error = AudioValidator.validate_file(audio_path)
        if not is_valid:
            return ErrorHandler.handle_audio_error(ValueError(error), audio_path)
        
        try:
            result = self.detector.predict(audio_path)
            result['success'] = True
            return result
        except Exception as e:
            return ErrorHandler.handle_model_error(e, self.detector.model_type)
