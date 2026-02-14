"""
API REST pour la classification de biens de consommation
CLF04 - Place de marché

FastAPI application for product classification using text and optional images.
"""

import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration des chemins
API_DIR = Path(__file__).resolve().parent
PROJECT_DIR = API_DIR.parent
MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data"

# Chemins des modèles
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
TEXT_MODEL_PATH = MODEL_DIR / "logistic_regression_model.pkl"

# Configuration de l'API
app = FastAPI(
    title="CLF04 - API de Classification de Biens",
    description="""
    API REST pour la classification automatique de produits e-commerce.
    
    ## Fonctionnalités
    
    * **Classification textuelle**: Prédiction à partir du nom, description et marque
    * **Probabilités**: Retourne le top-K des catégories avec confiance
    * **Métadonnées**: Informations détaillées sur les prédictions
    * **Santé**: Endpoint de monitoring
    
    ## Utilisation
    
    1. Envoyez une description produit à `/predict`
    2. Recevez la catégorie prédite avec niveau de confiance
    3. Consultez les alternatives via le top-K
    """,
    version="2.0.0",
    contact={
        "name": "Master IA - DIT",
        "email": "support@example.com"
    },
    license_info={
        "name": "Academic Project"
    }
)

# Configuration CORS pour permettre les appels depuis frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Modèles Pydantic =====

class ProductText(BaseModel):
    """Modèle pour une description produit."""
    product_name: Optional[str] = Field(None, description="Nom du produit", example="Nike Running Shoes")
    description: str = Field(..., description="Description détaillée en anglais", example="Comfortable running shoes with breathable mesh upper")
    brand: Optional[str] = Field(None, description="Marque du produit", example="Nike")

    class Config:
        json_schema_extra = {
            "example": {
                "product_name": "Nike Air Zoom Pegasus",
                "description": "Premium running shoes for men with comfortable sole, black color, breathable mesh upper, perfect for jogging and fitness",
                "brand": "Nike"
            }
        }


class PredictionResult(BaseModel):
    """Résultat d'une prédiction."""
    category: str = Field(..., description="Catégorie prédite")
    confidence: float = Field(..., description="Niveau de confiance (0-1)", ge=0, le=1)
    rank: int = Field(..., description="Rang de la prédiction", ge=1)


class PredictResponse(BaseModel):
    """Réponse complète de prédiction."""
    success: bool = Field(..., description="Statut de la requête")
    prediction: str = Field(..., description="Catégorie principale prédite")
    confidence: float = Field(..., description="Confiance de la prédiction principale")
    top_k_predictions: List[PredictionResult] = Field(..., description="Top K prédictions")
    metadata: Dict = Field(..., description="Métadonnées de la prédiction")
    timestamp: str = Field(..., description="Horodatage de la prédiction")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "prediction": "Clothing >> Men's Clothing >> Shoes >> Sports Shoes",
                "confidence": 0.78,
                "top_k_predictions": [
                    {"category": "Clothing >> Men's Clothing >> Shoes >> Sports Shoes", "confidence": 0.78, "rank": 1},
                    {"category": "Clothing >> Footwear >> Running Shoes", "confidence": 0.12, "rank": 2}
                ],
                "metadata": {
                    "text_length": 125,
                    "n_classes": 413,
                    "processing_time_ms": 45.2
                },
                "timestamp": "2026-02-13T10:30:00"
            }
        }


class BatchPredictRequest(BaseModel):
    """Requête pour prédictions par lot."""
    products: List[ProductText] = Field(..., description="Liste de produits à classifier")
    top_k: int = Field(5, description="Nombre de prédictions à retourner", ge=1, le=10)

    class Config:
        json_schema_extra = {
            "example": {
                "products": [
                    {
                        "product_name": "Nike Running Shoes",
                        "description": "Comfortable running shoes",
                        "brand": "Nike"
                    }
                ],
                "top_k": 5
            }
        }


class BatchPredictResponse(BaseModel):
    """Réponse pour prédictions par lot."""
    success: bool
    predictions: List[PredictResponse]
    total_processed: int
    total_time_ms: float


class HealthResponse(BaseModel):
    """Réponse du health check."""
    status: str = Field(..., description="Statut de l'API")
    models_loaded: bool = Field(..., description="Modèles chargés")
    version: str = Field(..., description="Version de l'API")
    timestamp: str = Field(..., description="Horodatage")
    model_info: Optional[Dict] = Field(None, description="Informations sur les modèles")


class ErrorResponse(BaseModel):
    """Réponse en cas d'erreur."""
    success: bool = False
    error: str = Field(..., description="Message d'erreur")
    detail: Optional[str] = Field(None, description="Détails de l'erreur")
    timestamp: str = Field(..., description="Horodatage")


# ===== Cache des modèles =====

class ModelCache:
    """Cache pour les modèles ML."""
    def __init__(self):
        self.vectorizer = None
        self.text_model = None
        self.loaded = False
        self.load_timestamp = None
    
    def load(self):
        """Charge les modèles en mémoire."""
        if self.loaded:
            return
        
        try:
            logger.info("Chargement des modèles...")
            start_time = time.time()
            
            # Charger le vectorizer
            with open(VECTORIZER_PATH, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info(f"✓ Vectorizer chargé: {self.vectorizer.max_features} features")
            
            # Charger le modèle
            with open(TEXT_MODEL_PATH, 'rb') as f:
                self.text_model = pickle.load(f)
            logger.info(f"✓ Modèle chargé: {len(self.text_model.classes_)} classes")
            
            self.loaded = True
            self.load_timestamp = datetime.now()
            
            load_time = (time.time() - start_time) * 1000
            logger.info(f"✓ Modèles chargés en {load_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des modèles: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Impossible de charger les modèles: {str(e)}"
            )
    
    def get_models(self):
        """Retourne les modèles (charge si nécessaire)."""
        if not self.loaded:
            self.load()
        return self.vectorizer, self.text_model
    
    def get_info(self):
        """Retourne les informations sur les modèles."""
        if not self.loaded:
            return None
        
        return {
            "vectorizer": {
                "type": type(self.vectorizer).__name__,
                "max_features": self.vectorizer.max_features,
                "ngram_range": self.vectorizer.ngram_range
            },
            "classifier": {
                "type": type(self.text_model).__name__,
                "n_classes": len(self.text_model.classes_),
                "classes_sample": self.text_model.classes_[:3].tolist()
            },
            "loaded_at": self.load_timestamp.isoformat() if self.load_timestamp else None
        }


# Instance globale du cache
model_cache = ModelCache()


# ===== Fonctions utilitaires =====

def combine_text_features(product: ProductText) -> str:
    """Combine les features textuelles en une seule chaîne."""
    parts = []
    if product.product_name:
        parts.append(product.product_name)
    if product.description:
        parts.append(product.description)
    if product.brand:
        parts.append(product.brand)
    return " ".join(parts).strip()


def predict_category(text: str, top_k: int = 5) -> Dict:
    """Prédit la catégorie d'un texte."""
    start_time = time.time()
    
    # Récupérer les modèles
    vectorizer, clf = model_cache.get_models()
    
    # Vectoriser le texte
    X = vectorizer.transform([text])
    
    # Prédire
    prediction = clf.predict(X)[0]
    probas = clf.predict_proba(X)[0]
    
    # Top K prédictions
    top_k_idx = np.argsort(probas)[-top_k:][::-1]
    top_k_classes = [clf.classes_[i] for i in top_k_idx]
    top_k_probas = [float(probas[i]) for i in top_k_idx]
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "prediction": prediction,
        "confidence": float(probas[top_k_idx[0]]),
        "top_k_classes": top_k_classes,
        "top_k_probas": top_k_probas,
        "processing_time_ms": processing_time,
        "text_length": len(text),
        "n_classes": len(clf.classes_)
    }


# ===== Endpoints =====

@app.on_event("startup")
async def startup_event():
    """Chargement des modèles au démarrage."""
    logger.info("🚀 Démarrage de l'API CLF04...")
    try:
        model_cache.load()
        logger.info("✓ API prête")
    except Exception as e:
        logger.error(f"❌ Erreur au démarrage: {e}")


@app.get("/", tags=["General"])
async def root():
    """Page d'accueil de l'API."""
    return {
        "message": "API de Classification de Biens de Consommation - CLF04",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "batch_predict": "POST /batch-predict",
            "health": "GET /health",
            "models": "GET /models/info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Vérification de l'état de santé de l'API.
    
    Retourne le statut de l'API et des informations sur les modèles chargés.
    """
    try:
        models_info = model_cache.get_info()
        
        return HealthResponse(
            status="healthy" if model_cache.loaded else "degraded",
            models_loaded=model_cache.loaded,
            version="2.0.0",
            timestamp=datetime.now().isoformat(),
            model_info=models_info
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            models_loaded=False,
            version="2.0.0",
            timestamp=datetime.now().isoformat(),
            model_info=None
        )


@app.get("/models/info", tags=["Models"])
async def get_models_info():
    """
    Retourne les informations sur les modèles chargés.
    
    Utile pour le debugging et le monitoring.
    """
    if not model_cache.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèles non chargés"
        )
    
    return {
        "success": True,
        "models": model_cache.get_info(),
        "paths": {
            "vectorizer": str(VECTORIZER_PATH),
            "classifier": str(TEXT_MODEL_PATH)
        }
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_product(
    product: ProductText,
    top_k: int = 5
):
    """
    Prédit la catégorie d'un produit à partir de sa description.
    
    **Paramètres:**
    - **product_name**: Nom du produit (optionnel)
    - **description**: Description détaillée en anglais (requis)
    - **brand**: Marque du produit (optionnel)
    - **top_k**: Nombre de prédictions à retourner (défaut: 5)
    
    **Retourne:**
    - Catégorie prédite avec niveau de confiance
    - Top K prédictions alternatives
    - Métadonnées de la prédiction
    
    **Exemple:**
    ```json
    {
        "product_name": "Nike Running Shoes",
        "description": "Comfortable running shoes with breathable mesh",
        "brand": "Nike"
    }
    ```
    """
    try:
        # Combiner les features textuelles
        text = combine_text_features(product)
        
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Au moins un champ texte doit être fourni"
            )
        
        # Prédire
        result = predict_category(text, top_k)
        
        # Construire la réponse
        top_k_predictions = [
            PredictionResult(
                category=cat,
                confidence=conf,
                rank=i + 1
            )
            for i, (cat, conf) in enumerate(zip(result["top_k_classes"], result["top_k_probas"]))
        ]
        
        return PredictResponse(
            success=True,
            prediction=result["prediction"],
            confidence=result["confidence"],
            top_k_predictions=top_k_predictions,
            metadata={
                "text_length": result["text_length"],
                "n_classes": result["n_classes"],
                "processing_time_ms": round(result["processing_time_ms"], 2)
            },
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


@app.post("/batch-predict", response_model=BatchPredictResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictRequest):
    """
    Prédit les catégories pour plusieurs produits en une seule requête.
    
    **Paramètres:**
    - **products**: Liste de produits à classifier
    - **top_k**: Nombre de prédictions par produit
    
    **Retourne:**
    - Liste des prédictions pour chaque produit
    - Statistiques globales
    
    **Exemple:**
    ```json
    {
        "products": [
            {
                "product_name": "Nike Shoes",
                "description": "Running shoes",
                "brand": "Nike"
            },
            {
                "description": "Cotton bedsheet with floral design"
            }
        ],
        "top_k": 3
    }
    ```
    """
    try:
        start_time = time.time()
        predictions = []
        
        for product in request.products:
            try:
                # Réutiliser l'endpoint /predict
                pred = await predict_product(product, request.top_k)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Erreur sur un produit: {e}")
                # Continuer avec les autres produits
                continue
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchPredictResponse(
            success=True,
            predictions=predictions,
            total_processed=len(predictions),
            total_time_ms=round(total_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du batch predict: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du traitement par lot: {str(e)}"
        )


@app.post("/predict/simple", tags=["Prediction"])
async def simple_predict(
    text: str = Form(..., description="Description du produit en anglais")
):
    """
    Endpoint simplifié pour prédiction rapide avec un seul champ texte.
    
    **Paramètres:**
    - **text**: Description du produit (form data)
    
    **Retourne:**
    - Catégorie prédite
    - Niveau de confiance
    
    Utile pour tester rapidement l'API via curl ou formulaire HTML.
    
    **Exemple curl:**
    ```bash
    curl -X POST "http://localhost:8000/predict/simple" \\
         -F "text=Nike running shoes for men"
    ```
    """
    try:
        result = predict_category(text, top_k=1)
        
        return {
            "success": True,
            "category": result["prediction"],
            "confidence": result["confidence"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur simple predict: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# ===== Gestion des erreurs =====

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handler personnalisé pour les erreurs HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handler pour les erreurs non capturées."""
    logger.error(f"Erreur non gérée: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Erreur interne du serveur",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ===== Démarrage de l'API =====

if __name__ == "__main__":
    logger.info("🚀 Démarrage du serveur API...")
    uvicorn.run(
        "api_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
