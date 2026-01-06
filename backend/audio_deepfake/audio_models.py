import tensorflow as tf
from tensorflow.keras.applications import MobileNet, VGG16, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.models import Model
import os


def build_mobilenet_model():
    """
    Construire un modèle MobileNet pour la détection de deepfake audio
    Entrée: Spectrogramme (224, 224, 3)
    Sortie: Probabilités [Real, Fake]
    """
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Geler les poids de base pour le fine-tuning
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)  # 2 classes: REAL, FAKE
    
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def build_vgg16_model():
    """
    Construire un modèle VGG16 pour la détection de deepfake audio
    Entrée: Spectrogramme (224, 224, 3)
    Sortie: Probabilités [Real, Fake]
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Geler les poids de base pour le fine-tuning
    base_model.trainable = False
    
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(2, activation='softmax')(x)  # 2 classes: REAL, FAKE
    
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def build_resnet_model():
    """
    Construire un modèle ResNet50 pour la détection de deepfake audio
    Entrée: Spectrogramme (224, 224, 3)
    Sortie: Probabilités [Real, Fake]
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Geler les poids de base pour le fine-tuning
    base_model.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(2, activation='softmax')(x)  # 2 classes: REAL, FAKE
    
    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def load_trained_model(model_type, weights_path):
    """
    Charger un modèle avec les poids entraînés
    
    Args:
        model_type (str): Type du modèle ('mobilenet', 'vgg16', 'resnet50')
        weights_path (str): Chemin vers le fichier de poids
        
    Returns:
        model: Modèle TensorFlow avec poids chargés
    """
    if model_type == 'mobilenet':
        model = build_mobilenet_model()
    elif model_type == 'vgg16':
        model = build_vgg16_model()
    elif model_type == 'resnet50':
        model = build_resnet_model()
    else:
        raise ValueError(f"Type de modèle non reconnu: {model_type}")
    
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        raise FileNotFoundError(f"Fichier de poids non trouvé: {weights_path}")
    
    return model


def get_available_models():
    """Obtenir la liste des modèles disponibles"""
    return ['mobilenet', 'vgg16', 'resnet50']
