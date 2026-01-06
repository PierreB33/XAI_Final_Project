"""
Configuration et constantes pour le système de détection de deepfake audio
"""

AUDIO_MODELS = ['mobilenet', 'vgg16', 'resnet50']

XAI_METHODS = ['grad_cam', 'lime', 'shap']

AUDIO_PREPROCESSING = {
    'target_sr': 22050,  
    'duration': None,    
    'n_mels': 128,       
    'fmax': 8000,       
}


SPECTROGRAM_PARAMS = {
    'figsize': (10, 4),
    'dpi': 100,
    'mel': True,  
}

MODEL_INPUT = {
    'image_size': (224, 224),
    'num_channels': 3,
    'normalization': 'image_net'  
}

XAI_PARAMS = {
    'grad_cam': {
        'class_index': 1,  
    },
    'lime': {
        'num_samples': 1000,
        'num_features': 10,
    },
    'shap': {
        'background_samples': 100,
        'sample_size': 50,
    }
}

PATHS = {
    'models': './models',
    'datasets': './datasets',
    'results': './results',
    'temp': './temp',
}

CLASS_MAPPING = {
    0: 'REAL',
    1: 'FAKE'
}


PRETRAINED_WEIGHTS = {
    'mobilenet': None,  
    'vgg16': None,    
    'resnet50': None,  
}

XAI_COMPATIBILITY = {
    'audio': ['grad_cam', 'lime', 'shap'],
    'image': ['grad_cam', 'lime', 'shap'],
}

CONFIDENCE_THRESHOLDS = {
    'high': 0.85,
    'medium': 0.65,
    'low': 0.5,
}

TF_ENABLE_ONEDNN_OPTS=0
