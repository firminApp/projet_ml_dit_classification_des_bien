# CLF04 - Classification des biens de consommation

Projet Réalisé par le Groupe 4 : josias AHOGA, Kpakpou BANIGATE, DOGO Amine et DIOUF Abdoulaye

## C'est quoi ce projet ?

On se met dans la peau d'un Data Scientist qui travaille pour une marketplace e-commerce ("Place de marche"). Le probleme c'est que les vendeurs categorisent mal leurs produits quand ils les postent, du coup on veut automatiser ca avec du ML.

On a un dataset Flipkart (site e-commerce indien) avec 1050 produits repartis dans 7 categories. L'idee c'est de voir si on peut predire la bonne categorie juste a partir de la description textuelle du produit.

## Ce qu'on a fait

1. **Exploration des donnees** : on a regarde les stats de base, les valeurs manquantes, la distribution des categories, des prix, etc.
2. **Pretraitement du texte** : nettoyage (minuscules, suppression ponctuation, stop words...) puis vectorisation avec TF-IDF
3. **Entrainement de 7 modeles** : Naive Bayes, Logistic Regression, SVM, Random Forest, KNN, Decision Tree, Gradient Boosting
4. **Optimisation** : GridSearchCV sur les 2 meilleurs modeles
5. **Sauvegarde** : modeles sauvegardes avec joblib (.pkl)
6. **API** : une interface Streamlit pour tester les predictions

## Structure du repo

```
├── README.md                              # vous etes ici
├── requirements.txt                       # les dependances a installer
├── .gitignore
├── classification_biens_consommation.ipynb # le notebook principal
├── data/
│   └── flipkart_com-ecommerce_sample_1050.csv
├── models/                                # generes apres execution du notebook
│   ├── final_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   └── label_encoder.pkl
├── api/
│   ├── app.py                             # l'interface Streamlit
│   ├── requirements.txt
│   └── README.md
└── outputs/                               # les graphiques generes
```

## Resultats

En gros le SVM lineaire et la regression logistique marchent tres bien (~97% d'accuracy). Le texte seul suffit pour classifier les produits correctement dans la grande majorite des cas.

| Modele              | Accuracy (test) | F1 Score |
|---------------------|-----------------|----------|
| Linear SVM          | ~0.97           | ~0.97    |
| Logistic Regression | ~0.97           | ~0.97    |
| Naive Bayes         | ~0.95           | ~0.95    |
| Random Forest       | ~0.93           | ~0.93    |
| Gradient Boosting   | ~0.93           | ~0.93    |
| KNN                 | ~0.89           | ~0.89    |
| Decision Tree       | ~0.84           | ~0.84    |

> Les scores exacts sont dans le notebook, ici c'est arrondi.

## Comment lancer le projet

### Installer les dependances

```bash
pip install -r requirements.txt
```

### Executer le notebook

Ouvrir `classification_biens_consommation.ipynb` avec Jupyter et executer toutes les cellules. Ca va generer les modeles dans `models/` et les graphiques dans `outputs/`.

### Lancer l'interface Streamlit

```bash
streamlit run api/app.py
```

Ensuite aller sur `http://localhost:8501` pour tester les predictions.

## Points importants

- On a bien separe train/test **avant** tout pretraitement pour eviter le data leakage
- L'imputation (mediane, mode) est calculee sur le train uniquement puis appliquee au test
- Le TF-IDF est fit sur le train, transform sur le test
- Le dataset est equilibre (150 produits par categorie) donc pas besoin de techniques de resampling

## Ce qu'on pourrait ameliorer

- Tester avec plus de donnees (1050 c'est quand meme petit)
- Utiliser les images des produits (avec un CNN)
- Essayer des modeles de deep learning type BERT pour le texte
- Parser la colonne `product_specifications` qui contient des infos utiles mais dans un format complique

## Outils utilises

Python, pandas, numpy, scikit-learn, matplotlib, seaborn, Streamlit, joblib

---

