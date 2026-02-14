# 🎯 Guide d'Utilisation - App Streamlit Optimisée

## ⚡ Démarrage Rapide

### 1. Créer les Modèles Optimisés (Première utilisation)

```bash
# Option A: Exécuter le script de training
python train_best_model.py

# Option B: Utiliser le notebook (cellules 88-100)
# Ouvrir classification_biens.ipynb et exécuter:
# - Cellule 88: Filtrage des classes (OBLIGATOIRE EN PREMIER)
# - Cellules 90-100: Autres améliorations
```

### 2. Lancer l'Application

```bash
streamlit run app/streamlit_app.py
```

Ouvrir dans le navigateur: `http://localhost:8501`

## 🆕 Nouvelles Fonctionnalités

### 1. Champ Prix
- **Impact**: Améliore la précision pour les catégories sensibles au prix
- **Format**: Nombre décimal (ex: 3999.0)
- **Optionnel**: Mais recommandé

### 2. Champ Marque
- **Impact**: Aide à différencier les produits similaires
- **Format**: Texte libre (ex: "Nike", "Samsung")
- **Optionnel**: Mais recommandé

### 3. Indicateur de Modèle
- 🚀 **Modèle Optimisé**: Précision ~69% (si optimized_*.pkl existent)
- ⚠️ **Modèle Baseline**: Précision ~21% (fallback)

## 📊 Exemple Complet

### Input
```
Nom: Nike Air Max 90
Description: Nike running shoes for men with air cushioning, 
             comfortable mesh upper, durable rubber sole, 
             perfect for sports and casual wear
Marque: Nike
Prix: 3999.0
```

### Output
```
🎯 Catégorie prédite: Footwear >> Men's Footwear >> Sports Shoes
Confiance: 92.5%

📊 Top 5:
1. Footwear >> Men's Footwear >> Sports Shoes (92.5%)
2. Footwear >> Men's Footwear >> Casual Shoes (5.2%)
3. Footwear >> Running Shoes (1.8%)
...
```

## 🔄 Différences Modèle Optimisé vs Baseline

| Feature | Optimisé | Baseline |
|---------|----------|----------|
| **Accuracy** | 69.4% ✅ | 21% ⚠️ |
| **Catégories** | 56 (filtrées) | 642 (toutes) |
| **Features TF-IDF** | 2475 (trigrams) | 5000 (bigrams) |
| **Features numériques** | + Prix, Discount, Marque | ❌ |
| **Équilibrage** | class_weight='balanced' | ❌ |
| **Temps de prédiction** | ~0.1s | ~0.1s |

## 💡 Conseils d'Utilisation

### ✅ À Faire
- Descriptions détaillées en anglais
- Ajouter prix et marque quand possible
- Utiliser les exemples prédéfinis pour tester
- Vérifier le modèle actif dans la sidebar

### ❌ À Éviter
- Descriptions trop courtes ("shoes")
- Mélanger français et anglais
- Prix négatifs ou irréalistes
- Texte avec trop de fautes

## 🐛 Résolution de Problèmes

### "Utilisation du modèle de base (précision: ~21%)"
➡️ Les modèles optimisés n'existent pas. Exécutez:
```bash
python train_best_model.py
```

### Prédictions incohérentes
➡️ Vérifiez que vous utilisez le modèle optimisé (indicateur en haut)

### Erreur au démarrage
➡️ Vérifiez que les dépendances sont installées:
```bash
pip install streamlit pandas numpy scikit-learn scipy
```

## 📈 Impact des Features

### Prix (+5-10% accuracy)
- Aide pour: Electronics, Footwear, Fashion
- Important pour distinguer: Premium vs Budget

### Marque (+3-8% accuracy)  
- Aide pour: Tous les produits de marque
- Top marques reconnues: Nike, Samsung, Adidas, etc.

### Description (+40% accuracy baseline)
- Essentiel, toujours requis
- Plus c'est détaillé, mieux c'est

## 🎯 Cas d'Usage

### E-commerce
- Auto-catégorisation de nouveaux produits
- Validation des catégories existantes
- Suggestions de catégories alternatives

### Marketplace
- Aide aux vendeurs pour catégoriser
- Contrôle qualité des listings
- Recherche et filtrage améliorés

---

**Version**: 2.0.0  
**Dernière mise à jour**: Février 2026
