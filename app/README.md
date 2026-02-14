# Application Streamlit - Classification de Biens de Consommation

Application web interactive pour la classification automatique de produits e-commerce à partir de leur description textuelle ou de leur image.

## 🚀 Installation

### Prérequis
- Python 3.11 ou supérieur
- Les modèles entraînés dans le dossier `../models/`

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## ▶️ Lancement de l'application

Depuis le dossier `app/` :

```bash
streamlit run streamlit_app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

## 📋 Fonctionnalités

### 1. Classification Textuelle 📝
- Entrez la description d'un produit en anglais
- Obtenez la catégorie prédite avec un niveau de confiance
- Visualisez les 5 meilleures prédictions
- Exemples prédéfinis pour tester rapidement

### 2. Classification par Image 🖼️
- Téléchargez une image de produit (JPG, JPEG, PNG)
- Extraction automatique des features avec ResNet50
- Classification basée sur les caractéristiques visuelles
- ⚠️ Nécessite TensorFlow et le modèle multimodal

### 3. Informations ℹ️
- Détails du projet
- Statistiques et métriques
- Technologies utilisées
- Guide d'utilisation

## 🎯 Utilisation

### Classification textuelle

1. Allez dans l'onglet "📝 Classification Textuelle"
2. Entrez le nom du produit (optionnel)
3. Saisissez une description détaillée en anglais
4. Ajoutez la marque si disponible (optionnel)
5. Cliquez sur "🔍 Classifier"
6. Consultez la catégorie prédite et les alternatives

**Exemple :**
```
Nom: Nike Running Shoes
Description: Nike running shoes for men, comfortable sole, black color, breathable mesh upper, perfect for jogging and fitness
Marque: Nike
```

### Classification par image

1. Allez dans l'onglet "🖼️ Classification par Image"
2. Cliquez sur "Browse files" ou glissez-déposez une image
3. Visualisez l'aperçu de l'image
4. Cliquez sur "🔍 Classifier l'image"
5. Les features visuelles seront extraites

**Note :** La classification complète par image nécessite le modèle multimodal entraîné.

## 🛠️ Architecture

```
app/
├── streamlit_app.py       # Application principale
├── requirements.txt       # Dépendances Python
└── README.md             # Ce fichier

../models/                # Modèles entraînés
├── tfidf_vectorizer.pkl
└── logistic_regression_model.pkl

../data/Images/           # Images des produits
```

## 📊 Modèles utilisés

### Modèle Texte
- **Vectorisation :** TF-IDF (5000 features, bigrammes)
- **Classificateur :** Logistic Regression
- **Performance :** Voir notebook pour métriques détaillées

### Modèle Image
- **Extracteur :** ResNet50 pré-entraîné (ImageNet)
- **Features :** 2048 dimensions
- **État :** Extraction de features disponible

## 🐛 Dépannage

### Erreur : Modèles non trouvés
```
❌ Erreur lors du chargement des modèles
```
**Solution :** Vérifiez que les fichiers suivants existent :
- `../models/tfidf_vectorizer.pkl`
- `../models/logistic_regression_model.pkl`

### TensorFlow non disponible
```
⚠️ TensorFlow n'est pas installé
```
**Solution :** Installez TensorFlow :
```bash
pip install tensorflow==2.16.0
```

### Erreur NumPy
```
AttributeError: np.complex_ was removed
```
**Solution :** Utilisez NumPy 1.26.4 :
```bash
pip install numpy==1.26.4
```

## 🚀 Déploiement

### Streamlit Cloud

1. Commitez le code sur GitHub
2. Connectez-vous sur [streamlit.io/cloud](https://streamlit.io/cloud)
3. Déployez depuis votre repository
4. Configurez les secrets si nécessaire

### Heroku

```bash
# Créez un Procfile
echo "web: streamlit run streamlit_app.py --server.port=$PORT" > Procfile

# Déployez
heroku create
git push heroku main
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]
```

## 📝 Notes

- L'application charge les modèles en cache pour de meilleures performances
- Les prédictions sont instantanées après le premier chargement
- Le dataset contient 642 catégories différentes
- Pour de meilleurs résultats, fournissez des descriptions détaillées

## 🔄 Améliorations futures

- [ ] Implémenter le modèle multimodal complet (texte + image)
- [ ] Ajouter une API REST
- [ ] Supporter plusieurs langues
- [ ] Améliorer l'interface utilisateur
- [ ] Ajouter des graphiques interactifs
- [ ] Permettre le téléchargement par lot

## 📧 Support

Pour toute question ou problème, consultez la documentation du projet ou ouvrez une issue sur GitHub.

---

**Version :** 1.0.0  
**Date :** Février 2026  
**Framework :** Streamlit
