#!/bin/bash

# Script de lancement de l'API FastAPI
# Classification de biens de consommation - CLF04

echo "🚀 Lancement de l'API CLF04..."
echo ""

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "api_app.py" ]; then
    echo "❌ Erreur: api_app.py non trouvé"
    echo "   Assurez-vous d'exécuter ce script depuis le dossier api/"
    exit 1
fi

# Vérifier que les modèles existent
if [ ! -d "../models" ]; then
    echo "⚠️  Attention: Dossier models/ non trouvé"
    echo "   Les modèles doivent être dans ../models/"
fi

if [ ! -f "../models/tfidf_vectorizer.pkl" ]; then
    echo "⚠️  Attention: tfidf_vectorizer.pkl non trouvé"
fi

if [ ! -f "../models/logistic_regression_model.pkl" ]; then
    echo "⚠️  Attention: logistic_regression_model.pkl non trouvé"
fi

echo ""
echo "📊 Configuration:"
echo "   Host: 0.0.0.0"
echo "   Port: 8000"
echo "   Reload: Activé"
echo ""
echo "📚 Documentation disponible sur:"
echo "   - Swagger UI: http://localhost:8000/docs"
echo "   - ReDoc: http://localhost:8000/redoc"
echo ""
echo "🔍 Endpoints principaux:"
echo "   - GET  /health"
echo "   - POST /predict"
echo "   - POST /batch-predict"
echo "   - GET  /models/info"
echo ""

# Lancer l'API
python api_app.py

# Si l'API se ferme ou échoue
echo ""
echo "✋ API arrêtée"
