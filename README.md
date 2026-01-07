# XAI FINAL PROJECT


## Introduction et contexte
Cette plateforme unifie deux verticales existantes d’Explainable AI : (1) détection de deepfake audio à partir de spectrogrammes et (2) détection de cancer du poumon sur radiographies thoraciques. Elle propose une interface unique pour déposer un média (audio/image), sélectionner un modèle compatible, choisir une ou plusieurs méthodes d’explicabilité (LIME, Grad-CAM, SHAP) et comparer les sorties. Les objectifs clés sont :
- Support multi-modal obligatoire (audio .wav et images radio thoraciques).
- Filtrage automatique des modèles et méthodes XAI selon le type de média.
- Visualisation côte à côte pour analyser plusieurs explications sur un même input.
- Expérience utilisateur simple (drag-and-drop, sélection guidée) et documentation claire pour la démo.

## Generative AI Usage Statement
We used Generative AI (LLM-based assistants) in the following ways for this project:

- System design: drafted and refined the unified architecture (FastAPI backend, TensorFlow/PyTorch pipelines, React/Vite frontend) with AI suggestions, then validated choices within the team.
- Code drafting: generated initial versions of many functions and modules (both audio and image flows), followed by human review, debugging, and testing before integration; some functions were written manually.
- Documentation: produced the first draft of the technical review using AI, then performed human corrections and adjustments.
- Guidance: consulted AI for next-step decisions and option trade-offs throughout development; final implementation choices were made by the team.

All AI-assisted outputs were reviewed and adjusted by the team before inclusion in the submission.

## Technologies et modèles
- Backend: FastAPI (Python) avec TensorFlow/Keras pour l’audio et PyTorch/torchvision (ResNet50) pour l’image.
- Modèles audio: MobileNet, VGG16, ResNet50 opérant sur spectrogrammes 224x224.
- Modèle image: ResNet50, poids personnalisés optionnels ou ImageNet par défaut.
- XAI audio: Grad-CAM, LIME, SHAP sur spectrogrammes.
- XAI image: Grad-CAM, LIME (SHAP non disponible pour l’image dans cette version).
- Frontend: React + Vite (TanStack Router) pour l’UI d’upload, sélection de modèle/XAI et restitution des résultats.

## Mode d’emploi (à compléter)
- Pré-requis, installation et lancement (API + frontend).
- Parcours utilisateur: upload audio/image, sélection modèle, sélection méthodes XAI, visualisation des résultats.
- Notes de compatibilité (filtres XAI par modalité) et bonnes pratiques d’utilisation.

