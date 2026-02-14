# API REST - Classification de Biens de Consommation

API REST FastAPI pour la classification automatique de produits e-commerce.

## 🚀 Installation

### Prérequis
- Python 3.11 ou supérieur
- Les modèles entraînés dans `../models/`

### Installation des dépendances

```bash
cd api
pip install -r requirements.txt
```

## ▶️ Lancement de l'API

### Développement

```bash
cd api
python api_app.py
```

Ou avec uvicorn directement:

```bash
uvicorn api_app:app --reload --host 0.0.0.0 --port 8000
```

L'API sera disponible sur:
- **API**: http://localhost:8000
- **Documentation (Swagger)**: http://localhost:8000/docs
- **Documentation alternative (ReDoc)**: http://localhost:8000/redoc

### Production

```bash
gunicorn api_app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📚 Documentation

La documentation interactive est automatiquement générée et disponible sur:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔌 Endpoints

### 1. Health Check

**GET** `/health`

Vérifie l'état de santé de l'API.

```bash
curl http://localhost:8000/health
```

**Réponse:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "2.0.0",
  "timestamp": "2026-02-13T10:30:00",
  "model_info": {
    "vectorizer": {
      "type": "TfidfVectorizer",
      "max_features": 5000,
      "ngram_range": [1, 2]
    },
    "classifier": {
      "type": "LogisticRegression",
      "n_classes": 413
    }
  }
}
```

### 2. Prédiction Simple

**POST** `/predict`

Prédit la catégorie d'un produit.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Nike Running Shoes",
    "description": "Comfortable running shoes with breathable mesh upper",
    "brand": "Nike"
  }'
```

**Paramètres:**
- `product_name` (string, optionnel): Nom du produit
- `description` (string, requis): Description en anglais
- `brand` (string, optionnel): Marque
- `top_k` (int, optionnel): Nombre de prédictions (défaut: 5)

**Réponse:**
```json
{
  "success": true,
  "prediction": "Clothing >> Men's Clothing >> Shoes >> Sports Shoes",
  "confidence": 0.78,
  "top_k_predictions": [
    {
      "category": "Clothing >> Men's Clothing >> Shoes >> Sports Shoes",
      "confidence": 0.78,
      "rank": 1
    },
    {
      "category": "Clothing >> Footwear >> Running Shoes",
      "confidence": 0.12,
      "rank": 2
    }
  ],
  "metadata": {
    "text_length": 125,
    "n_classes": 413,
    "processing_time_ms": 45.2
  },
  "timestamp": "2026-02-13T10:30:00"
}
```

### 3. Prédiction Simple (Form Data)

**POST** `/predict/simple`

Version simplifiée pour test rapide.

```bash
curl -X POST "http://localhost:8000/predict/simple" \
  -F "text=Nike running shoes for men black color"
```

**Réponse:**
```json
{
  "success": true,
  "category": "Clothing >> Men's Clothing >> Shoes",
  "confidence": 0.65,
  "timestamp": "2026-02-13T10:30:00"
}
```

### 4. Prédiction par Lot

**POST** `/batch-predict`

Classifie plusieurs produits en une requête.

```bash
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Réponse:**
```json
{
  "success": true,
  "predictions": [
    { /* prédiction 1 */ },
    { /* prédiction 2 */ }
  ],
  "total_processed": 2,
  "total_time_ms": 89.5
}
```

### 5. Informations sur les Modèles

**GET** `/models/info`

Retourne les détails des modèles chargés.

```bash
curl http://localhost:8000/models/info
```

## 🧪 Tests

### Test rapide

```bash
cd api
python test_api.py
```

### Test avec curl

```bash
# Health check
curl http://localhost:8000/health

# Prédiction simple
curl -X POST "http://localhost:8000/predict/simple" \
  -F "text=Nike running shoes for men"

# Prédiction complète
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Cotton Bedsheet",
    "description": "Premium quality cotton bedsheet with floral design, king size, includes 2 pillow covers",
    "brand": "Elegance"
  }'
```

### Test avec Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Prédiction
data = {
    "product_name": "Nike Running Shoes",
    "description": "Comfortable running shoes with breathable mesh",
    "brand": "Nike"
}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## 📊 Performance

- **Temps de réponse moyen**: 40-60ms par prédiction
- **Chargement des modèles**: Au démarrage (< 1s)
- **Concurrent requests**: Support natif avec uvicorn
- **Mise en cache**: Modèles chargés une seule fois en mémoire

## 🔒 Sécurité

### CORS

Par défaut, l'API accepte les requêtes de toutes origines. En production, modifiez:

```python
allow_origins=["https://votredomaine.com"]
```

### Rate Limiting

Pour limiter le nombre de requêtes, ajoutez:

```bash
pip install slowapi
```

## 🚀 Déploiement

### Docker

```bash
docker build -t clf04-api .
docker run -p 8000:8000 clf04-api
```

### Heroku

```bash
heroku create clf04-api
git push heroku main
```

### AWS Lambda

Utilisez Mangum pour adapter FastAPI:

```bash
pip install mangum
```

## 📝 Logs

Les logs sont disponibles dans la console:

```
2026-02-13 10:30:00 - INFO - 🚀 Démarrage de l'API CLF04...
2026-02-13 10:30:01 - INFO - ✓ Vectorizer chargé: 5000 features
2026-02-13 10:30:01 - INFO - ✓ Modèle chargé: 413 classes
2026-02-13 10:30:01 - INFO - ✓ API prête
```

## 🐛 Dépannage

### Erreur: Modèles non trouvés

```
❌ Erreur lors du chargement des modèles
```

**Solution:** Vérifiez que les fichiers existent:
- `../models/tfidf_vectorizer.pkl`
- `../models/logistic_regression_model.pkl`

### Port déjà utilisé

```
ERROR: [Errno 48] Address already in use
```

**Solution:** Changez le port ou arrêtez le processus:

```bash
lsof -ti:8000 | xargs kill -9
```

### Erreur de version scikit-learn

```
InconsistentVersionWarning: Trying to unpickle estimator
```

**Solution:** Installez la bonne version:

```bash
pip install scikit-learn==1.7.1
```

## 📈 Monitoring

### Prometheus Metrics

Ajoutez:

```bash
pip install prometheus-fastapi-instrumentator
```

### Health Endpoint

Utilisez `/health` pour vérifier régulièrement le statut:

```bash
*/5 * * * * curl -f http://localhost:8000/health || alert
```

## 🔄 Mise à jour des Modèles

Pour mettre à jour les modèles:

1. Entraînez de nouveaux modèles
2. Sauvegardez dans `../models/`
3. Redémarrez l'API

Les modèles sont rechargés automatiquement au démarrage.

## 📚 Références

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Deployment](https://www.uvicorn.org/deployment/)
- [Pydantic Models](https://docs.pydantic.dev/)

## 👥 Support

Pour toute question ou problème:
- Consultez la documentation: http://localhost:8000/docs
- Vérifiez les logs
- Testez avec `/health`

---

**Version:** 2.0.0  
**License:** Academic Project  
**Python:** 3.11+
