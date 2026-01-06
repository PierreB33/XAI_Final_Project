"""
Lung Cancer Detection Backend - Configuration
"""

# Model configuration
MODEL_CONFIG = {
    'num_classes': 3,
    'input_size': 224,
    'class_names': ['Lung Adenocarcinoma', 'Lung Benign', 'Lung Squamous Cell Carcinoma'],
    'class_indices': {
        0: 'Lung Adenocarcinoma',
        1: 'Lung Benign',
        2: 'Lung Squamous Cell Carcinoma'
    }
}

# Image preprocessing configuration
PREPROCESSING_CONFIG = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'image_size': 224,
    'normalize': True
}

# Grad-CAM configuration
GRADCAM_CONFIG = {
    'layer_name': 'layer4',
    'layer_index': -1,
    'colormap': 'jet',
    'alpha': 0.5
}

# LIME configuration
LIME_CONFIG = {
    'num_samples': 1000,
    'num_features': 10,
    'hide_color': 0,
    'top_labels': 1
}

# Output configuration
OUTPUT_CONFIG = {
    'image_format': 'png',
    'dpi': 100,
    'quality': 95
}

# Device configuration
DEVICE_CONFIG = {
    'use_cuda': True,
    'num_workers': 0
}
