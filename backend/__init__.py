"""
Backend package pour la détection de deepfake audio avec XAI
"""

from .deepfake_detector import DeepfakeAudioDetector
from .xai_explainer import XAIExplainer
from .audio_models import build_mobilenet_model, build_vgg16_model, build_resnet_model, get_available_models
from .spectrogram_converter import create_spectrogram, batch_convert_wavs, get_audio_info
from .config import AUDIO_MODELS, XAI_METHODS, CLASS_MAPPING

__version__ = '1.0.0'
__author__ = 'XAI Team'
__all__ = [
    'DeepfakeAudioDetector',
    'XAIExplainer',
    'build_mobilenet_model',
    'build_vgg16_model',
    'build_resnet_model',
    'get_available_models',
    'create_spectrogram',
    'batch_convert_wavs',
    'get_audio_info',
    'AUDIO_MODELS',
    'XAI_METHODS',
    'CLASS_MAPPING',
]
