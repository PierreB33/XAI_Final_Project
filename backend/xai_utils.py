import tensorflow as tf
from tensorflow.keras.applications import MobileNet, VGG16, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.models import Model

def build_mobilenet_model():
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def build_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# Exemple de fonction pour charger les poids entraînés dans ton API
def load_trained_model(model_type, weights_path):
    if model_type == 'mobilenet':
        model = build_mobilenet_model()
    elif model_type == 'vgg16':
        model = build_vgg16_model()
    
    model.load_weights(weights_path)
    return model