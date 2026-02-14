#!/bin/bash

# Script de lancement de l'application Streamlit
# Classification de biens de consommation - CLF04

echo "🚀 Lancement de l'application Streamlit..."
echo ""

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Erreur: streamlit_app.py non trouvé"
    echo "   Assurez-vous d'exécuter ce script depuis le dossier app/"
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

if [ ! -f "../models/final_model.pkl" ]; then
    echo "⚠️  Attention: final_model.pkl non trouvé"
fi

echo ""
echo "📊 Démarrage de Streamlit..."
echo ""

# Lancer Streamlit
streamlit run streamlit_app.py

# Si Streamlit se ferme ou échoue
echo ""
echo "✋ Application arrêtée"
