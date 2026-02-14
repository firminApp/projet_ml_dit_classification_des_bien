# Changelog - Application Streamlit

## Version 2.1.0 (14 février 2026)

### 🔄 Modifications Majeures

#### Suppression de la Classification par Image
- ❌ Supprimé l'onglet "Classification par Image"
- ❌ Supprimé la fonction `load_image_model()`
- ❌ Supprimé la fonction `extract_image_features()`
- ❌ Supprimé la dépendance PIL/Pillow pour les images
- ❌ Supprimé la dépendance TensorFlow/Keras

#### Simplification de l'Interface
- ✅ 2 onglets au lieu de 3:
  1. 📝 Classification Textuelle (inchangé)
  2. ℹ️ À propos (inchangé)

### 📊 Fonctionnalités Conservées

#### Classification Textuelle Optimisée
- ✅ Description + Nom du produit
- ✅ Prix (optionnel - améliore la précision)
- ✅ Marque (optionnel - améliore la précision)
- ✅ Top 5 des prédictions avec pourcentages
- ✅ Graphiques de probabilités
- ✅ Modèle optimisé (69.4% accuracy)

### 🎯 Raison des Changements

1. **Performance**: Élimination des dépendances lourdes (TensorFlow ~500MB)
2. **Simplicité**: Focus sur la classification textuelle qui fonctionne bien
3. **Rapidité**: Chargement plus rapide de l'application
4. **Maintenance**: Code plus simple et maintenable

### 📈 Impact

| Aspect | Avant | Après |
|--------|-------|-------|
| **Onglets** | 3 | 2 |
| **Dépendances** | 6 packages | 4 packages |
| **Taille install** | ~600 MB | ~100 MB |
| **Temps chargement** | ~10s | ~2s |
| **Fonctionnalités** | Texte + Image (non fonctionnelle) | Texte |

### 🚀 Utilisation

```bash
# Lancer l'application
streamlit run app/streamlit_app.py
```

### 📦 Dépendances Requises

```txt
streamlit
pandas
numpy
scikit-learn
scipy
```

### 🔮 Roadmap Future

- [ ] Deep Learning pour texte (BERT, Transformers)
- [ ] Classification multimodale (quand modèle disponible)
- [ ] API REST dédiée
- [ ] Déploiement cloud

---

## Version 2.0.0 (13 février 2026)

### Ajouts
- ✅ Modèles optimisés avec feature engineering
- ✅ Support prix et marque
- ✅ Amélioration +231% accuracy (21% → 69.4%)
- ✅ Filtrage des classes (642 → 56)

---

**Développé par**: Équipe Data Science  
**Projet**: CLF04 - Classification de Biens  
**Dernière mise à jour**: 14 février 2026
