"""
Backend package pour la détection de deepfake audio avec XAI
"""

from .audio_deepfake.deepfake_detector import DeepfakeAudioDetector
from .audio_deepfake.xai_explainer import XAIExplainer
from .audio_deepfake.audio_models import build_mobilenet_model, build_vgg16_model, build_resnet_model, get_available_models
from .audio_deepfake.spectrogram_converter import create_spectrogram, batch_convert_wavs, get_audio_info
from .audio_deepfake.config import AUDIO_MODELS, XAI_METHODS, CLASS_MAPPING

__version__ = '1.0.0'
__author__ = ''
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
