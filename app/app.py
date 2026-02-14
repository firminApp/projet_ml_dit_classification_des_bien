"""
Application Streamlit : Classification des biens de consommation
Projet CLF04 :Place de Marche E-commerce

Interface web pour predire la categorie d'un produit
a partir de sa description textuelle.
"""

import os
import re
import string
import joblib
import streamlit as st

# ============================================================
# Configuration de la page
# ============================================================

st.set_page_config(
    page_title="Classification des biens de consommation",
    page_icon="🛒",
    layout="centered"
)

# ============================================================
# Chargement du modele (avec cache pour eviter de recharger)
# ============================================================

# Chemin vers le dossier models (au niveau parent)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(PARENT_DIR, "models"))


@st.cache_resource
def load_model():
    """Charger le modele et les artefacts une seule fois."""
    try:
        model_path = os.path.join(MODEL_DIR, "final_model.pkl")
        tfidf_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
        label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
        
        # Afficher les chemins pour debug
        st.info(f"Chargement depuis : {MODEL_DIR}")
        
        if not os.path.exists(model_path):
            st.error(f"Fichier introuvable : {model_path}")
            return None, None, None
            
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        label_encoder = joblib.load(label_encoder_path)
        
        st.success("✅ Modèles chargés avec succès !")
        return model, tfidf, label_encoder
    except FileNotFoundError as e:
        st.error(f"❌ Erreur : fichier introuvable - {e}")
        st.info(f"Dossier models attendu : {MODEL_DIR}")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement : {e}")
        return None, None, None


model, tfidf, label_encoder = load_model()


# ============================================================
# Stop words (meme liste que dans le notebook)
# ============================================================

STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
    'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
    'will', 'just', 'don', 'should', 'now', 'also', 'would', 'could', 'may', 'shall',
    'nan', 'none', 'product', 'buy', 'price', 'online', 'best', 'shop', 'india',
    'free', 'delivery', 'flipkart', 'available', 'offer', 'discount', 'sale'
}


# ============================================================
# Fonction de nettoyage du texte (identique au notebook)
# ============================================================

def clean_text(text):
    """Nettoyer et preprocesser le texte."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(tokens)


# ============================================================
# Interface
# ============================================================

st.title("🛒 Classification des biens de consommation")
st.markdown(
    "Cette application predit la categorie d'un produit e-commerce "
    "a partir de sa description textuelle."
)

# Sidebar avec les infos
with st.sidebar:
    st.header("A propos")
    st.markdown(
        "**Projet CLF04** - Place de Marche E-commerce\n\n"
        "Modele : LinearSVC optimise\n\n"
        "Accuracy : 94.76%\n\n"
        "Vectorisation : TF-IDF (5000 features)"
    )

    st.header("Categories disponibles")
    if label_encoder is not None:
        for cat in label_encoder.classes_:
            st.markdown(f"- {cat}")

# Zone de saisie
st.subheader("Entrez une description de produit")

# Exemples pre-remplis
exemples = {
    "-- Choisir un exemple --": "",
    "Montre Samsung": "Samsung Galaxy Watch 42mm Bluetooth smartwatch with heart rate monitor",
    "Ordinateur HP": "HP Pavilion 15 laptop Intel Core i5 8GB RAM 512GB SSD Windows 11",
    "Shampoing bebe": "Organic baby shampoo gentle tear-free formula 200ml for sensitive skin",
    "Ustensile cuisine": "Stainless steel pressure cooker 5 liters induction compatible with lid",
    "Creme visage": "Anti-aging face cream with vitamin C and hyaluronic acid moisturizer",
    "Decoration": "LED fairy string lights for Diwali decoration 10 meters waterproof",
    "Linge de maison": "Cotton bedsheet king size floral print with 2 pillow covers"
}

exemple_choisi = st.selectbox("Ou selectionnez un exemple :", list(exemples.keys()))

# Text area avec l'exemple pre-rempli si selectionne
default_text = exemples[exemple_choisi]
text_input = st.text_area(
    "Description du produit (en anglais) :",
    value=default_text,
    height=100,
    placeholder="Ex: Samsung Galaxy Watch 42mm Bluetooth smartwatch..."
)

# Bouton de prediction
if st.button("Predire la categorie", type="primary"):
    if not text_input.strip():
        st.warning("Veuillez entrer une description de produit.")
    elif model is None:
        st.error("Le modele n'est pas charge. Verifiez le dossier models/.")
    else:
        # Nettoyage
        cleaned = clean_text(text_input)

        # Verification que le texte nettoye n'est pas vide
        if not cleaned.strip():
            st.warning("Le texte nettoye est vide. Essayez avec plus de mots descriptifs.")
        else:
            # Vectorisation
            vectorized = tfidf.transform([cleaned])
            
            # Prediction de la categorie principale
            prediction_enc = model.predict(vectorized)[0]
            category = label_encoder.inverse_transform([prediction_enc])[0]

            # Calcul des probabilites/scores de decision si disponible
            if hasattr(model, 'decision_function'):
                # Pour SVM et Logistic Regression
                scores = model.decision_function(vectorized)[0]
                # Obtenir les indices des top-5 scores
                top_indices = scores.argsort()[-5:][::-1]
                top_categories = label_encoder.inverse_transform(top_indices)
                top_scores = scores[top_indices]
                # Normaliser les scores pour affichage (softmax-like)
                import numpy as np
                exp_scores = np.exp(top_scores - np.max(top_scores))
                probas = exp_scores / exp_scores.sum()
            elif hasattr(model, 'predict_proba'):
                # Pour les modeles avec probabilites
                probas_all = model.predict_proba(vectorized)[0]
                top_indices = probas_all.argsort()[-5:][::-1]
                top_categories = label_encoder.inverse_transform(top_indices)
                probas = probas_all[top_indices]
            else:
                # Fallback : juste la prediction principale
                top_categories = [category]
                probas = [1.0]

            # Affichage du resultat principal
            st.success(f"**Categorie predite : {category}**")
            
            # Affichage du score de confiance
            if len(probas) > 0:
                confidence = probas[0] * 100
                st.info(f"**Confiance : {confidence:.2f}%**")
            
            # Top 5 des predictions
            st.subheader("Top 5 des categories probables :")
            for i, (cat, prob) in enumerate(zip(top_categories, probas), 1):
                percentage = prob * 100
                st.write(f"{i}. **{cat}** - {percentage:.2f}%")
                st.progress(float(prob))

            # Details dans un expander
            with st.expander("Voir les details du traitement"):
                st.markdown(f"**Texte original :** {text_input}")
                st.markdown(f"**Texte nettoye :** {cleaned}")
                st.markdown(f"**Nombre de mots nettoyes :** {len(cleaned.split())}")
                st.markdown(f"**Nombre de features TF-IDF :** {vectorized.shape[1]}")
                st.markdown(f"**Type de modele :** {type(model).__name__}")
