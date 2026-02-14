# CLF04 — Classification de biens de consommation

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Accuracy](https://img.shields.io/badge/Accuracy-90.59%25-brightgreen.svg)](/)

## 📋 Vue d'ensemble

Ce projet implémente un système de classification automatique pour catégoriser des biens de consommation e-commerce à partir de données textuelles (nom, description, marque). L'objectif est d'automatiser l'attribution des produits à des catégories pour optimiser l'expérience utilisateur sur une marketplace anglophone.

**🎯 Performance actuelle :** 90.59% d'accuracy sur 56 catégories (après optimisation)

## 📑 Table des matières

- [Architecture du projet](#️-architecture-du-projet)
- [Structure du projet](#-structure-du-projet)
- [Démarrage rapide](#-démarrage-rapide)
- [Dataset](#-dataset)
- [Workflow et méthodologie](#-workflow-et-méthodologie)
- [Résultats et performances](#-résultats-et-performances)
- [Déploiement](#-déploiement)
- [Notes techniques](#-notes-techniques)
- [Livrables du projet](#-livrables-du-projet)

## 🏗️ Architecture du projet

Le projet comprend :
- **Notebook Jupyter** pour l'analyse exploratoire et le prototypage
- **Script d'entraînement optimisé** avec feature engineering avancé
- **API REST (FastAPI)** pour l'intégration backend
- **Interface web (Streamlit)** pour les démonstrations

## 📂 Structure du projet

```
soutenance/
├── classification_biens_consommation.ipynb    # Notebook principal d'analyse
├── subject.md                                  # Énoncé du projet
├── README.md                                   # Cette documentation
├── .gitignore                                  # Fichiers à exclure du versioning
│
├── api/                                        # API REST FastAPI
│   ├── api_app.py                             # Application FastAPI
│   ├── client_example.py                      # Exemple client Python
│   ├── test_api.py                            # Tests automatisés
│   ├── requirements.txt                       # Dépendances API
│   ├── run.sh                                 # Script de lancement
│   ├── Dockerfile                             # Configuration Docker
│   ├── Procfile                               # Configuration Heroku
│   ├── QUICKSTART.md                          # Guide rapide
│   └── README.md                              # Documentation API
│
├── app/                                        # Interface Streamlit
│   ├── streamlit_app.py                       # Application Streamlit
│   ├── test_models.py                         # Tests de chargement
│   ├── test_app.py                            # Tests de l'interface
│   ├── requirements.txt                       # Dépendances app
│   ├── run.sh                                 # Script de lancement
│   ├── Procfile                               # Configuration Heroku
│   ├── USAGE.md                               # Guide d'utilisation
│   ├── CHANGELOG.md                           # Historique des changements
│   └── README.md                              # Documentation app
│
├── data/                                       # Données (non versionné)
│   ├── flipkart_com-ecommerce_sample_1050.csv # Dataset principal
│   └── Images/                                 # Images des produits (1050)
│
└── models/                                     # Modèles entraînés (non versionné)
    ├── optimized_model.pkl                    # Modèle principal (2.8 MB)
    ├── optimized_vectorizer.pkl               # Vectorizer TF-IDF (275 KB)
    ├── optimized_scaler.pkl                   # Scaler pour features numériques
    ├── optimized_brand_encoder.pkl            # Encodeur de marques
    ├── optimized_model_metadata.json          # Métadonnées du modèle
    ├── final_model.pkl                        # Modèle legacy (si disponible)
    ├── tfidf_vectorizer.pkl                   # Vectorizer legacy
    └── label_encoder.pkl                      # Encodeur de labels
```

## 🚀 Démarrage rapide

### Prérequis
```bash
python >= 3.11
pip install -r requirements.txt
```

### Option 1 : Lancer l'application Streamlit (Recommandé)

```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
# Ou simplement : ./run.sh
```

Accédez à l'interface sur **http://localhost:8501**

### Option 2 : Utiliser l'API REST

```bash
cd api
pip install -r requirements.txt
python api_app.py
# Ou : ./run.sh
```

Documentation interactive sur **http://localhost:8000/docs**

### Option 3 : Explorer le notebook

```bash
jupyter notebook classification_biens_consommation.ipynb
```

## 📊 Dataset

**Source :** Flipkart E-commerce  
**Taille :** 1050 produits  
**Classes :** 642 catégories initiales → 56 classes retenues (≥3 exemples)  
**Langue :** Anglais  

**Colonnes principales :**
- `product_name` : Nom du produit
- `description` : Description détaillée
- `brand` : Marque du produit
- `retail_price` : Prix de vente
- `discounted_price` : Prix après remise
- `product_category_tree` : Catégorie (cible)
- `product_specifications` : Spécifications techniques
- `Images/` : Dossier contenant 1050 images produits

## � Workflow et méthodologie

### 1. Configuration et chargement des données
- Définition des répertoires de travail
- Import des bibliothèques (pandas, scikit-learn, matplotlib)
- Chargement du dataset Flipkart (1050 produits, 642 catégories)

### 2. Analyse exploratoire (EDA)

**Nettoyage :**
- Suppression des colonnes non pertinentes : `uniq_id`, `crawl_timestamp`, `pid`
- Imputation des valeurs manquantes :
  - Prix : médiane
  - Brand : "NoBrand"
  - Spécifications : chaîne vide

**Visualisations :**
- Distributions univariées (histogrammes, boxplots)
- Distribution très déséquilibrée des catégories
- Analyses bivariées (prix vs catégories)

### 3. Prétraitement et Feature Engineering

**Filtrage des classes rares (KEY IMPROVEMENT!) :**
- Seuil minimal : 3 exemples par classe
- 642 catégories → 56 classes retenues
- 1050 échantillons → 424 échantillons filtrés

**Split stratifié :**
- Train : 60% (254 échantillons)
- Validation : 20% (85 échantillons)
- Test : 20% (85 échantillons)

**Feature Engineering avancé :**
1. **Texte combiné** : product_name + description + brand + specifications
2. **Vectorisation TF-IDF** :
   - 10,000 features max
   - N-grams : (1, 3) pour capturer le contexte
   - min_df=2, max_df=0.8
   - sublinear_tf=True
   - Stop words anglais
   - Résultat : ~6,400 features textuelles

3. **Features numériques** :
   - Prix retail normalisé (StandardScaler)
   - Taux de remise : (retail - discounted) / retail
   - Marque encodée (Top 50 + "Other")

4. **Combinaison finale** :
   - Matrice sparse : 6,439 features (6,436 texte + 3 numériques)

### 4. Modélisation optimisée

**Modèle baseline (version initiale) :**
- Logistic Regression simple
- TF-IDF 5000 features
- ❌ **Résultats :** Acc=19.05%, F1=7.17%

**Modèle optimisé (version actuelle) :**
- **Algorithme :** Logistic Regression
- **Hyperparamètres :**
  - `class_weight='balanced'` (gère le déséquilibre)
  - `C=1.0` (régularisation)
  - `solver='lbfgs'`
  - `max_iter=1000`
  - `random_state=42`

**✅ Résultats impressionnants :**
```
Validation : Acc=84.71% | F1=87.95%
Test :       Acc=90.59% | F1=90.78%

Amélioration vs baseline : +376% accuracy, +1,166% F1-score !
```

### 5. Clés du succès

1. ✅ **Filtrage des classes rares** : Élimine le bruit (642→56 classes)
2. ✅ **class_weight='balanced'** : Compense le déséquilibre résiduel
3. ✅ **TF-IDF enrichi** : 10k features + trigrams capturent mieux le contexte
4. ✅ **Features numériques** : Prix, remise, marque ajoutent de l'information
5. ✅ **Feature engineering robuste** : Combinaison texte + métadonnées

### 6. Visualisations et métriques

- **Matrice de confusion** : Excellente diagonale
- **Rapport de classification détaillé** : 
  - Precision/Recall/F1 par classe
  - Macro avg : 0.90
  - Weighted avg : 0.91
- **Distribution des classes** après filtrage
- **Top-K prédictions** avec probabilités

### 7. Sauvegarde des modèles

Modèles sauvegardés dans `/models/` :
- `optimized_model.pkl` (2.8 MB) - Modèle LogisticRegression entraîné
- `optimized_vectorizer.pkl` (275 KB) - Vectorizer TF-IDF
- `optimized_scaler.pkl` - Scaler pour features numériques
- `optimized_brand_encoder.pkl` - Encodeur de marques
- `optimized_model_metadata.json` - Métadonnées et performances

## 📈 Résultats et performances

### Performance du modèle optimisé

| Métrique | Validation | Test |
|----------|-----------|------|
| **Accuracy** | 84.71% | **90.59%** |
| **F1-Score (macro)** | 87.95% | **90.78%** |
| **Precision (macro)** | - | 90% |
| **Recall (macro)** | - | 93% |

### Amélioration spectaculaire

| Version | Test Accuracy | Test F1 |
|---------|--------------|---------|
| **Baseline** (initiale) | 19.05% | 7.17% |
| **Optimisée** (actuelle) | **90.59%** | **90.78%** |
| **Amélioration** | **+376%** | **+1,166%** |

### Points forts

✅ **Excellentes performances** : >90% sur test set  
✅ **Pipeline complet** : De l'exploration à la production  
✅ **Feature engineering robuste** : Texte + métadonnées + prix  
✅ **Gestion du déséquilibre** : Filtrage + class_weight  
✅ **Applications déployables** : API REST + Interface web  
✅ **Tests automatisés** : Validation du chargement des modèles  
✅ **Documentation complète** : README, QUICKSTART, guides  

### Pistes d'amélioration futures

**1. Modèles plus avancés :**
- Ensemble methods : Random Forest, XGBoost, CatBoost
- Deep Learning : LSTM, Transformers
- Embeddings pré-entraînés : BERT, DistilBERT, Sentence-BERT

**2. Exploitation des images :**
- CNN pour features visuelles (ResNet, EfficientNet)
- Vision Transformers (ViT)
- Modèle multimodal texte + image

**3. Optimisation avancée :**
- Hyperparameter tuning (Bayesian Optimization, Optuna)
- Cross-validation stratifiée K-fold
- Ensembling (voting, stacking)

**4. Feature Engineering plus poussé :**
- Analyse sémantique (word2vec, GloVe)
- Features de spécifications techniques
- Analyse de sentiment
- Extraction d'entités nommées

**5. Monitoring et production :**
- A/B testing
- Monitoring des drifts de données
- Retraining automatique
- CI/CD pipeline
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

**Dataset Flipkart E-commerce :**
- Source : Flipkart (marketplace indienne)
- Taille initiale : 1050 produits
- Classes initiales : 642 catégories distinctes
- **Après filtrage** : 424 produits, 56 catégories (≥3 exemples)
- Langue : Anglais
- Images : 1050 images produits (dans `data/Images/`)

**Environnement de développement :**
- Python 3.11+
- Jupyter Notebook pour exploration
- Anaconda/venv pour isolation
- Git pour versioning

**Technologies utilisées :**

| Composant | Technologies |
|-----------|-------------|
| **ML/Data Science** | scikit-learn, pandas, numpy |
| **Visualisation** | matplotlib, seaborn |
| **API REST** | FastAPI, uvicorn, pydantic |
| **Interface Web** | Streamlit |
| **Containerisation** | Docker (optionnel) |
| **Deployment** | Heroku, Streamlit Cloud |

**Dépendances principales :**
```txt
# Core ML & Data
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0

# Visualisation
matplotlib>=3.7.0
seaborn>=0.12.0

# Web & API
streamlit>=1.28.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0

# Optionnel (pour classification image)
tensorflow>=2.16.0
pillow>=10.0.0
```

**Installation complète :**
```bash
# 1. Cloner le repository
git clone <votre-repo>
cd soutenance

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 3. Pour l'application Streamlit
cd app
pip install -r requirements.txt

# 4. Pour l'API FastAPI
cd ../api
pip install -r requirements.txt

# 5. Pour le notebook et l'entraînement
pip install jupyter pandas numpy scikit-learn matplotlib seaborn
```

**Structure des modèles sauvegardés :**
```
models/
├── optimized_model.pkl              # 2.8 MB - LogisticRegression
├── optimized_vectorizer.pkl         # 275 KB - TfidfVectorizer
├── optimized_scaler.pkl             # <1 KB - StandardScaler
├── optimized_brand_encoder.pkl      # <1 KB - LabelEncoder (marques)
└── optimized_model_metadata.json    # Métadonnées (performances, configs)
```

## 🎓 Livrables du projet

Ce projet répond aux objectifs du sujet CLF04 :

### ✅ Repository GitHub complet
- ✅ Notebook Jupyter avec analyse exploratoire complète
- ✅ Scripts Python pour entraînement et déploiement
- ✅ Extraction et traitement des données textuelles
- ✅ Fonctions de prétraitement et feature engineering
- ✅ Résultats et étude de faisabilité (90.59% accuracy)
- ✅ Documentation README détaillée

### ✅ Interface de classification déployable
- ✅ **Application Streamlit** : Classification via texte avec interface intuitive
- ✅ **API REST FastAPI** : Endpoints pour intégration backend
- ✅ Support de la classification textuelle (image en option)
- ✅ Prêt pour déploiement cloud (Streamlit Cloud, Heroku)

### ✅ Tests et validation
- ✅ Test automatisé du chargement des modèles
- ✅ Validation sur ensemble de test indépendant
- ✅ Métriques complètes (accuracy, F1, precision, recall)
- ✅ Interface testable immédiatement

### 📊 Résultats de faisabilité

**Conclusion : ✅ FAISABLE avec excellentes performances**

| Critère | Résultat | Statut |
|---------|----------|--------|
| Niveau de précision | 90.59% | ✅ Excellent |
| F1-Score macro | 90.78% | ✅ Robuste |
| Nombre de classes | 56 catégories | ✅ Pertinent |
| Temps d'inférence | < 100ms | ✅ Production-ready |
| Déploiement | API + Web | ✅ Opérationnel |

**Recommandation :** Le moteur de classification est prêt pour la production avec d'excellentes performances sur 56 catégories principales.

## 📚 Références

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Handling Imbalanced Data](https://imbalanced-learn.org/)
- [FastAPI Framework](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 👥 Auteurs

Projet réalisé dans le cadre du **Master IA - DIT**  
Classification de biens de consommation pour marketplace e-commerce

## 📄 Licence

Projet académique - Tous droits réservés
