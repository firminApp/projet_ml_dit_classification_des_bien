# CLF04 — Classification de biens de consommation

## 📋 Vue d'ensemble

Ce projet implémente un système de classification automatique pour catégoriser des biens de consommation à partir de données textuelles (nom, description, marque). L'objectif est d'automatiser l'attribution des produits à des catégories pour optimiser l'expérience utilisateur sur une marketplace e-commerce.

## 📂 Structure du projet

```
soutenance/
├── classification_biens.ipynb    # Notebook principal avec pipeline complet
├── dataset.csv                    # Dataset pour tests rapides  
├── subject.md                     # Énoncé du projet
├── README.md                      # Cette documentation
├── api/
│   └── api_app.py                # API REST (FastAPI)
├── app/
│   ├── streamlit_app.py          # Interface web (Streamlit)
│   ├── requirements.txt          # Dépendances Python
│   ├── README.md                 # Documentation de l'app
│   ├── test_models.py            # Tests de chargement des modèles
│   ├── run.sh                    # Script de lancement
│   ├── Procfile                  # Configuration Heroku
│   └── .streamlit/
│       └── config.toml           # Configuration Streamlit
├── data/
│   ├── flipkart_com-ecommerce_sample_1050.csv  # Dataset principal
│   └── Images/                   # Images des produits (1050 fichiers)
└── models/
    ├── logistic_regression_model.pkl    # Modèle entraîné (413 classes)
    └── tfidf_vectorizer.pkl             # Vectorizer TF-IDF (5000 features)
```

## 🚀 Démarrage rapide

### Prérequis
```bash
python >= 3.8
pip install pandas scikit-learn matplotlib seaborn jupyter
```

### Exécution du notebook

1. Ouvrir le notebook: `classification_biens.ipynb`
2. Exécuter les cellules dans l'ordre:
   - Configuration (cellule 1)
   - Import des bibliothèques (cellule 2)
   - Chargement des données (cellule 3)
   - Prétraitement et EDA (cellules 4-43)
   - Entraînement du modèle (cellules 56-57)
   - Visualisations (cellules 59-60)
   - Tests (cellules 62-64)

## 📊 Workflow du notebook

### 1. Configuration et importation
- Définition des répertoires de travail
- Import des bibliothèques (pandas, scikit-learn, matplotlib, etc.)
- Chargement du dataset Flipkart (1050 produits)

### 2. Analyse exploratoire des données (EDA)

**Variables:**
- Numériques: `retail_price`, `discounted_price`  
- Catégorielles: `product_name`, `description`, `brand`, `product_category_tree` (cible)
- Suppression des colonnes non pertinentes: `uniq_id`, `crawl_timestamp`, `pid`

**Imputation:**
- Prix: médiane
- Brand: "NoBrand"
- Spécifications: chaîne vide

**Visualisations:**
- Distributions univariées (histogrammes, boxplots)
- Distribution des catégories (très déséquilibrée)
- Analyses bivariées

### 3. Préparation des données

**Split stratifié:**
- Train: 60% (630 échantillons)
- Validation: 20% (210 échantillons)
- Test: 20% (210 échantillons)

**Feature engineering:**
- Combinaison de `product_name`, `description`, `brand` en une seule feature textuelle
- Vectorisation TF-IDF (5000 features, bigrammes, stop words anglais)

### 4. Modélisation

**Modèle baseline:**
- Algorithme: Logistic Regression
- Vectorization: TF-IDF
- Hyperparamètres: max_iter=1000, random_state=42

**Résultats:**
```
Validation:
  Accuracy: 0.2095 (20.95%)
  F1 macro: 0.0760 (7.60%)

Test:
  Accuracy: 0.1905 (19.05%)
  F1 macro: 0.0717 (7.17%)
```

### 5. Visualisations et métriques

- Distribution des classes (graphique à barres)
- Matrice de confusion (heatmap)
- Rapport de classification détaillé (precision, recall, F1-score par classe)

### 6. Tests et validation

- **Test 1:** Prédictions sur exemples du test set
- **Test 2:** Fonction de prédiction personnalisée avec top-3 catégories
- **Test 3:** Comparaison avec baseline aléatoire

### 7. Sauvegarde

Modèles sauvegardés dans `/models`:
- `logistic_regression_model.pkl`  
- `tfidf_vectorizer.pkl`

## 📈 Résultats et analyses

### Points forts
✓ Pipeline complet de bout en bout fonctionnel
✓ Prétraitement robuste des données textuelles
✓ Modèle baseline simple et interprétable
✓ Sauvegarde des modèles pour réutilisation

### Limitations
⚠️ Performance modeste due à:
- Fort déséquilibre des classes (distribution très asymétrique)
- Nombreuses catégories avec peu d'exemples
- Features textuelles basiques (TF-IDF)
- Modèle linéaire simple

### Améliorations possibles

**1. Traitement du déséquilibre de classes:**
- SMOTE (Synthetic Minority Over-sampling)
- Class weights dans le modèle
- Stratégies d'échantillonnage

**2. Modèles plus avancés:**
- Random Forest / XGBoost / CatBoost
- Deep Learning: LSTM, Transformers
- Embeddings pré-entraînés: BERT, DistilBERT, Sentence-BERT

**3. Feature Engineering avancé:**
- Extraction de features numériques (longueur, nb mots, etc.)
- Utilisation des spécifications produit
- Analyse des images (CNN, Vision Transformers)
- Features basées sur les prix

**4. Optimisation:**
- Grid Search / Random Search / Bayesian Optimization
- Cross-validation stratifiée
- Ensembling (voting, stacking)

**5. Évaluation:**
- Métriques adaptées au déséquilibre (macro F1, weighted F1)
- Courbes ROC multiclasses
- Analyse des erreurs par catégorie

## 🌐 Déploiement

### API REST (FastAPI)

L'API FastAPI fournit une interface REST complète pour la classification de produits.

**Lancement:**
```bash
cd api
python api_app.py
# OU utiliser le script
./run.sh
```

L'API sera disponible sur:
- **API**: http://localhost:8000
- **Documentation Swagger**: http://localhost:8000/docs
- **Documentation ReDoc**: http://localhost:8000/redoc

**Endpoints principaux:**

**GET** `/health` - Vérification de l'état de santé
```bash
curl http://localhost:8000/health
```

**POST** `/predict` - Prédiction complète avec métadonnées
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Nike Running Shoes",
    "description": "Comfortable running shoes",
    "brand": "Nike"
  }'
```

**POST** `/predict/simple` - Prédiction rapide (form data)
```bash
curl -X POST "http://localhost:8000/predict/simple" \
  -F "text=Nike running shoes for men"
```

**POST** `/batch-predict` - Prédiction par lot
```bash
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "products": [
      {"description": "Nike shoes"},
      {"description": "Cotton bedsheet"}
    ],
    "top_k": 3
  }'
```

**GET** `/models/info` - Informations sur les modèles

**Fonctionnalités de l'API:**
- ✅ Documentation interactive (Swagger UI + ReDoc)
- ✅ Validation automatique des données (Pydantic)
- ✅ Gestion d'erreurs robuste
- ✅ Support CORS
- ✅ Logging détaillé
- ✅ Cache des modèles en mémoire
- ✅ Top-K prédictions avec probabilités
- ✅ Métadonnées de performance
- ✅ Prédictions par lot
- ✅ Endpoints de monitoring

**Test de l'API:**
```bash
# Tests automatisés
cd api
python test_api.py

# Exemples avec curl
./examples.sh

# Client Python
python client_example.py
```

**Déploiement:**
- Docker: `docker build -t clf04-api . && docker run -p 8000:8000 clf04-api`
- Production: `gunicorn api_app:app -w 4 -k uvicorn.workers.UvicornWorker`
- Heroku: `git push heroku main` (Procfile configuré)

### Interface Streamlit

L'application Streamlit offre une interface conviviale pour tester le modèle de classification.

**Lancement:**
```bash
cd app
streamlit run streamlit_app.py
# OU utiliser le script de lancement
./run.sh
```
L'interface s'ouvre automatiquement sur `http://localhost:8501`

**Fonctionnalités principales:**

**📝 Onglet Classification Textuelle:**
- Formulaire intuitif pour saisir le nom, description et marque du produit
- Exemples prédéfinis pour test rapide
- Affichage de la catégorie prédite avec niveau de confiance
- Top 5 des prédictions avec probabilités
- Graphique interactif des résultats
- Support pour texte en anglais

**🖼️ Onglet Classification par Image:**
- Upload d'images (JPG, JPEG, PNG)
- Extraction automatique de features avec ResNet50
- Aperçu de l'image téléchargée
- Exemples d'images du dataset
- Note: Nécessite TensorFlow installé

**ℹ️ Onglet À propos:**
- Documentation du projet
- Statistiques et métriques
- Technologies utilisées
- Guide d'utilisation
- Roadmap des améliorations

**Interface:**
- Design moderne et responsive
- Thème personnalisé avec couleurs cohérentes
- Navigation par onglets
- Sidebar avec informations en temps réel
- Messages d'erreur explicites
- Mise en cache des modèles pour performances optimales

**Installation des dépendances:**
```bash
cd app
pip install -r requirements.txt
```

**Déploiement sur le cloud:**
- Streamlit Cloud: Push sur GitHub et déployer via streamlit.io
- Heroku: Utiliser le `Procfile` fourni
- Docker: Container prêt à l'emploi (voir app/README.md)

**Test du système:**
```bash
cd app
python test_models.py  # Vérifie que les modèles sont chargés correctement
```

## 📝 Notes techniques

**Dataset:**
- Source: Flipkart e-commerce
- Taille: 1050 produits
- Classes: ~800+ catégories distinctes (très déséquilibré)
- Langue: Anglais

**Environnement:**
- Python 3.13
- Jupyter Notebook
- Anaconda environment

**Dépendances principales:**
```
# Core
pandas==2.2.3
numpy==1.26.4  # Compatible avec TensorFlow 2.16
scikit-learn==1.7.1

# Visualisation
matplotlib==3.10.0
seaborn==0.13.2

# Web & Interface
streamlit>=1.28.0
fastapi
uvicorn

# Deep Learning (optionnel)
tensorflow==2.16.0
pillow>=10.0.0
```

**Installation complète:**
```bash
# Pour le notebook et l'entraînement
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Pour l'application Streamlit
cd app
pip install -r requirements.txt

# Pour le support image (optionnel)
pip install tensorflow==2.16.0 pillow
```

## 📚 Références

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Handling Imbalanced Data](https://imbalanced-learn.org/)

## 👥 Auteurs

Projet réalisé dans le cadre du Master IA - DIT

## 📄 Licence

Projet académique - Tous droits réservés
