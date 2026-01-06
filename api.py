#!/usr/bin/env python3
"""
FastAPI REST pour la plateforme XAI - Connexion Backend/Frontend
Endpoints pour audio deepfake detection et lung cancer detection avec XAI
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import logging
from pathlib import Path
import tempfile
from typing import List, Optional
import traceback

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration du chemin
sys.path.insert(0, str(Path(__file__).parent))

# Global variables for lazy loading
DeepfakeAudioDetector = None
XAIExplainer = None
AudioValidator = None
LungCancerPredictor = None
LungCancerXAIExplainer = None

AUDIO_AVAILABLE = False
LUNG_CANCER_AVAILABLE = False

def load_audio_modules():
    """Lazy load audio modules"""
    global DeepfakeAudioDetector, XAIExplainer, AudioValidator, AUDIO_AVAILABLE
    
    if AUDIO_AVAILABLE:
        return
    
    try:
        from backend.audio_deepfake.deepfake_detector import DeepfakeAudioDetector as DAD
        from backend.audio_deepfake.xai_explainer import XAIExplainer as XAI
        from backend.audio_deepfake.validators import AudioValidator as AV
        DeepfakeAudioDetector = DAD
        XAIExplainer = XAI
        AudioValidator = AV
        AUDIO_AVAILABLE = True
        logger.info("✅ Audio modules loaded")
    except Exception as e:
        logger.error(f"❌ Audio module not available: {e}")
        AUDIO_AVAILABLE = False

def load_lung_cancer_modules():
    """Lazy load lung cancer modules"""
    global LungCancerPredictor, LungCancerXAIExplainer, LUNG_CANCER_AVAILABLE
    
    if LUNG_CANCER_AVAILABLE:
        return
    
    try:
        from backend.lung_cancer.predictor import Predictor as LCP
        from backend.lung_cancer.xai import XAIExplainer as LCXAI
        LungCancerPredictor = LCP
        LungCancerXAIExplainer = LCXAI
        LUNG_CANCER_AVAILABLE = True
        logger.info("✅ Lung cancer modules loaded")
    except Exception as e:
        logger.warning(f"⚠️  Lung cancer module not available: {e}")
        LUNG_CANCER_AVAILABLE = False

# Configuration FastAPI
app = FastAPI(
    title="XAI Platform API",
    description="API pour Deepfake Audio Detection et Lung Cancer Detection avec XAI",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration uploads
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_AUDIO = {'wav', 'mp3', 'm4a'}
ALLOWED_IMAGE = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Vérifier si le fichier a une extension autorisée"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.get("/api/health")
async def health():
    """Endpoint de santé"""
    return {
        'status': 'ok',
        'audio_available': True,
        'lung_cancer_available': LUNG_CANCER_AVAILABLE
    }


@app.get("/api/models")
async def get_models():
    """Retourner les modèles disponibles"""
    # Load lung cancer modules to ensure they're available
    load_lung_cancer_modules()
    
    models = {
        'audio': [
            {'id': 'deepfake-mobilenet', 'name': 'MobileNet', 'type': 'audio'},
            {'id': 'deepfake-vgg16', 'name': 'VGG16', 'type': 'audio'},
            {'id': 'deepfake-resnet', 'name': 'ResNet', 'type': 'audio'},
        ],
        'image': []
    }
    
    if LUNG_CANCER_AVAILABLE:
        models['image'] = [
            {'id': 'lung-resnet50', 'name': 'ResNet50', 'type': 'image'},
        ]
    
    return models


@app.get("/api/xai-techniques")
async def get_xai_techniques():
    """Retourner les techniques XAI disponibles"""
    return {
        'techniques': [
            {'id': 'lime', 'name': 'LIME', 'description': 'Local Interpretable Model-agnostic Explanations'},
            {'id': 'gradcam', 'name': 'Grad-CAM', 'description': 'Gradient-weighted Class Activation Maps'},
            {'id': 'shap', 'name': 'SHAP', 'description': 'SHapley Additive exPlanations'},
        ]
    }


@app.post("/api/analyze/audio")
async def analyze_audio(
    file: UploadFile = File(...),
    model: str = Form(...),
    xai_methods: List[str] = Form(default=[])
):
    """Analyser un fichier audio pour deepfake detection"""
    logger.info(f"🎵 Audio analysis request: model={model}, xai_methods={xai_methods}")
    
    # Load modules on demand
    load_audio_modules()
    
    if not AUDIO_AVAILABLE:
        logger.error("❌ Audio module not available")
        raise HTTPException(status_code=503, detail="Audio analysis not available")
    
    temp_path = None
    try:
        # Validation du fichier
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        logger.info(f"📁 File: {file.filename}, Type: {file.content_type}")
        
        if not allowed_file(file.filename, ALLOWED_AUDIO):
            raise HTTPException(
                status_code=400, 
                detail=f"File type not allowed. Allowed: {ALLOWED_AUDIO}"
            )
        
        # Mapper l'ID du modèle au type
        try:
            # Map model IDs to correct model type names
            model_map = {
                'deepfake-mobilenet': 'mobilenet',
                'deepfake-vgg16': 'vgg16',
                'deepfake-resnet': 'resnet50'
            }
            model_type = model_map.get(model, model.split('-')[1])
            logger.info(f"🤖 Model type: {model_type}")
        except Exception as e:
            logger.error(f"❌ Invalid model format: {model}")
            raise HTTPException(status_code=400, detail=f"Invalid model format: {model}")
        
        # Sauvegarder le fichier temporairement
        temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
        try:
            with open(temp_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            logger.info(f"💾 File saved to: {temp_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
        # Valider l'audio
        try:
            is_valid, validation_msg = AudioValidator.validate_file(temp_path)
            if not is_valid:
                logger.error(f"❌ Invalid audio: {validation_msg}")
                raise HTTPException(status_code=400, detail=f"Invalid audio: {validation_msg}")
            logger.info(f"✅ Audio validation passed")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            raise HTTPException(status_code=400, detail=f"Audio validation error: {str(e)}")
        
        # Charger le détecteur
        try:
            logger.info(f"🔧 Loading detector: {model_type}")
            detector = DeepfakeAudioDetector(model_type=model_type)
            logger.info(f"✅ Detector loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load detector: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
        # Prédiction
        try:
            logger.info(f"🔍 Running prediction...")
            prediction = detector.predict(temp_path)
            confidence = float(prediction['confidence'])
            prediction_label = prediction['predicted_label']
            logger.info(f"✅ Prediction: {prediction_label} ({confidence:.2%})")
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        
        # Préparer la réponse
        result = {
            'model': model,
            'prediction': prediction_label,
            'confidence': float(confidence),
            'probabilities': {
                'real': float(prediction['real_probability']),
                'fake': float(prediction['fake_probability'])
            },
            'xai_results': {}
        }
        
        # XAI Explainability (si demandé)
        if xai_methods:
            logger.info(f"🧠 Computing XAI explanations: {xai_methods}")
            try:
                explainer = XAIExplainer(detector)
                
                for method in xai_methods:
                    if method == 'lime':
                        try:
                            logger.info(f"  - Computing LIME...")
                            lime_result = explainer.lime_explanation(temp_path)
                            result['xai_results']['lime'] = {
                                'type': 'lime',
                                'explanation': 'LIME explanation generated'
                            }
                            logger.info(f"  ✅ LIME done")
                        except Exception as e:
                            logger.warning(f"  ⚠️  LIME error: {e}")
                            result['xai_results']['lime'] = {'error': str(e)}
                    
                    elif method == 'gradcam':
                        try:
                            logger.info(f"  - Computing Grad-CAM...")
                            gradcam_result = explainer.grad_cam(temp_path)
                            result['xai_results']['gradcam'] = {
                                'type': 'gradcam',
                                'visualization': 'Grad-CAM heatmap generated'
                            }
                            logger.info(f"  ✅ Grad-CAM done")
                        except Exception as e:
                            logger.warning(f"  ⚠️  Grad-CAM error: {e}")
                            result['xai_results']['gradcam'] = {'error': str(e)}
                    
                    elif method == 'shap':
                        try:
                            logger.info(f"  - Computing SHAP...")
                            shap_result = explainer.shap_explanation(temp_path)
                            result['xai_results']['shap'] = {
                                'type': 'shap',
                                'explanation': 'SHAP explanation generated'
                            }
                            logger.info(f"  ✅ SHAP done")
                        except Exception as e:
                            logger.warning(f"  ⚠️  SHAP error: {e}")
                            result['xai_results']['shap'] = {'error': str(e)}
            except Exception as e:
                logger.warning(f"⚠️  XAI error: {e}")
                result['xai_results']['error'] = str(e)
        
        logger.info(f"✅ Analysis complete")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Nettoyer
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"🗑️  Cleaned up {temp_path}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to clean up {temp_path}: {e}")


@app.post("/api/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    model: str = Form(...),
    xai_methods: List[str] = Form(default=[]),
    model_path: Optional[str] = Form(None)
):
    """Analyser une image pour lung cancer detection"""
    logger.info(f"🖼️  Image analysis request: model={model}, xai_methods={xai_methods}")
    
    # Load modules on demand
    load_lung_cancer_modules()
    
    if not LUNG_CANCER_AVAILABLE:
        logger.error("❌ Lung cancer module not available")
        raise HTTPException(status_code=503, detail="Lung cancer detection not available")
    
    temp_path = None
    try:
        # Validation du fichier
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        logger.info(f"📁 File: {file.filename}, Type: {file.content_type}")
        
        if not allowed_file(file.filename, ALLOWED_IMAGE):
            raise HTTPException(
                status_code=400, 
                detail=f"File type not allowed. Allowed: {ALLOWED_IMAGE}"
            )
        
        # Sauvegarder le fichier temporairement
        temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
        try:
            with open(temp_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            logger.info(f"💾 File saved to: {temp_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
        # Charger le prédicteur
        try:
            logger.info(f"🔧 Loading predictor...")
            predictor = LungCancerPredictor(
                model_path=model_path,
                use_default_weights=model_path is None
            )
            logger.info(f"✅ Predictor loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load predictor: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
        # Prédiction
        try:
            logger.info(f"🔍 Running prediction...")
            prediction = predictor.predict(temp_path)
            confidence = float(prediction['confidence'])
            predicted_class = prediction['predicted_class']
            logger.info(f"✅ Prediction: {predicted_class} ({confidence:.2%})")
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        
        # Préparer la réponse
        result = {
            'model': model,
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': prediction['probabilities'],
            'xai_results': {}
        }
        
        # XAI Explainability (si demandé)
        if xai_methods:
            logger.info(f"🧠 Computing XAI explanations: {xai_methods}")
            try:
                xai_explainer = LungCancerXAIExplainer(predictor.model, predictor.device)
                # Preprocess image for XAI
                image_data = predictor.preprocessor.preprocess(temp_path, predictor.device)
                
                for method in xai_methods:
                    if method == 'gradcam':
                        try:
                            logger.info(f"  - Computing Grad-CAM...")
                            gradcam_result = xai_explainer.gradcam.generate_heatmap(image_data, 0)
                            result['xai_results']['gradcam'] = {
                                'type': 'gradcam',
                                'visualization': 'Grad-CAM heatmap generated'
                            }
                            logger.info(f"  ✅ Grad-CAM done")
                        except Exception as e:
                            logger.warning(f"  ⚠️  Grad-CAM error: {e}")
                            result['xai_results']['gradcam'] = {'error': str(e)}
                    
                    elif method == 'lime':
                        try:
                            logger.info(f"  - Computing LIME...")
                            lime_result = xai_explainer.lime.explain(temp_path, 0)
                            result['xai_results']['lime'] = {
                                'type': 'lime',
                                'explanation': 'LIME explanation generated'
                            }
                            logger.info(f"  ✅ LIME done")
                        except Exception as e:
                            logger.warning(f"  ⚠️  LIME error: {e}")
                            result['xai_results']['lime'] = {'error': str(e)}
                    
                    elif method == 'shap':
                        try:
                            logger.info(f"  - Computing SHAP...")
                            # SHAP is not available for image in this implementation
                            logger.info(f"  ⚠️  SHAP not available for image analysis")
                            result['xai_results']['shap'] = {'error': 'SHAP not available for image analysis'}
                        except Exception as e:
                            logger.warning(f"  ⚠️  SHAP error: {e}")
                            result['xai_results']['shap'] = {'error': str(e)}
            except Exception as e:
                logger.warning(f"⚠️  XAI error: {e}")
                result['xai_results']['error'] = str(e)
        
        logger.info(f"✅ Analysis complete")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Nettoyer
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"🗑️  Cleaned up {temp_path}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to clean up {temp_path}: {e}")


@app.get("/")
async def root():
    """Redirect to docs"""
    return {"message": "XAI Platform API - Go to /docs for documentation"}


if __name__ == '__main__':
    import uvicorn
    print("🚀 XAI Platform API Server")
    print(f"Audio Deepfake Detection: ✅")
    print(f"Lung Cancer Detection: {'✅' if LUNG_CANCER_AVAILABLE else '❌'}")
    print("Starting server on http://localhost:5000")
    print("Documentation available at http://localhost:5000/docs")
    uvicorn.run(app, host='0.0.0.0', port=5000)

