# 🚀 Guide de Démarrage Rapide - API CLF04

Guide rapide pour démarrer avec l'API de classification.

## ⚡ Installation Express (5 minutes)

### Étape 1: Installer les dépendances

```bash
cd api
pip install fastapi uvicorn python-multipart
```

### Étape 2: Vérifier les modèles

```bash
ls -la ../models/
# Doit contenir:
# - tfidf_vectorizer.pkl
# - logistic_regression_model.pkl
```

### Étape 3: Démarrer l'API

```bash
python api_app.py
```

✅ L'API est prête sur http://localhost:8000

## 🧪 Test Rapide

Ouvrez un nouveau terminal:

```bash
# Health check
curl http://localhost:8000/health

# Prédiction simple
curl -X POST "http://localhost:8000/predict/simple" \
  -F "text=Nike running shoes for men"
```

## 📚 Documentation

Ouvrez dans votre navigateur:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

## 🐍 Test avec Python

```python
import requests

# Prédiction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "product_name": "Nike Shoes",
        "description": "Running shoes for men",
        "brand": "Nike"
    }
)

print(response.json())
```

## 📋 Commandes Utiles

```bash
# Démarrer l'API
python api_app.py

# Avec reload automatique
uvicorn api_app:app --reload

# Tests automatiques
python test_api.py

# Exemples curl
./examples.sh

# Client Python
python client_example.py
```

## 🎯 Endpoints Principaux

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | État de santé |
| POST | `/predict` | Prédiction complète |
| POST | `/predict/simple` | Prédiction simple |
| POST | `/batch-predict` | Prédiction par lot |
| GET | `/models/info` | Info modèles |

## 🔍 Exemples de Réponses

### Health Check
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "2.0.0"
}
```

### Prédiction
```json
{
  "success": true,
  "prediction": "Clothing >> Shoes >> Sports Shoes",
  "confidence": 0.78,
  "top_k_predictions": [
    {
      "category": "Clothing >> Shoes >> Sports Shoes",
      "confidence": 0.78,
      "rank": 1
    }
  ]
}
```

## ⚠️ Dépannage

### Port déjà utilisé
```bash
# Trouver et tuer le processus
lsof -ti:8000 | xargs kill -9
```

### Modèles non trouvés
Vérifiez les chemins dans `api_app.py`:
```python
MODEL_DIR = PROJECT_DIR / "models"
```

### Module non trouvé
```bash
pip install -r requirements.txt
```

## 🚀 Production

```bash
# Avec gunicorn (4 workers)
gunicorn api_app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Avec Docker
docker build -t clf04-api .
docker run -p 8000:8000 clf04-api
```

## 📖 Plus d'Informations

- **README complet**: `api/README.md`
- **Code source**: `api/api_app.py`
- **Tests**: `api/test_api.py`
- **Exemples**: `api/examples.sh`

---

✨ **Prêt à classifier!** L'API est maintenant opérationnelle.
