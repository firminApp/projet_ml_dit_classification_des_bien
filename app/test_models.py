"""
Test de chargement des modèles pour l'application Streamlit
"""
import pickle
from pathlib import Path

def test_model_loading():
    """Vérifie que les modèles peuvent être chargés"""
    
    APP_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = APP_DIR.parent
    MODEL_DIR = PROJECT_DIR / "models"
    
    print("🔍 Vérification des modèles...\n")
    
    # Vérifier les chemins
    print(f"📁 Dossier des modèles: {MODEL_DIR}")
    print(f"   Existe: {MODEL_DIR.exists()}\n")
    
    # Vérifier le vectorizer
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"
    print(f"📄 Vectorizer: {vectorizer_path.name}")
    print(f"   Existe: {vectorizer_path.exists()}")
    
    if vectorizer_path.exists():
        try:
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            print(f"   ✅ Chargé avec succès")
            print(f"   Type: {type(vectorizer).__name__}")
            if hasattr(vectorizer, 'max_features'):
                print(f"   Features: {vectorizer.max_features}")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    print()
    
    # Vérifier le modèle
    model_path = MODEL_DIR / "logistic_regression_model.pkl"
    print(f"📄 Modèle: {model_path.name}")
    print(f"   Existe: {model_path.exists()}")
    
    if model_path.exists():
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"   ✅ Chargé avec succès")
            print(f"   Type: {type(model).__name__}")
            if hasattr(model, 'classes_'):
                print(f"   Nombre de classes: {len(model.classes_)}")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    print()
    
    # Test de prédiction simple
    if vectorizer_path.exists() and model_path.exists():
        print("🧪 Test de prédiction...")
        try:
            test_text = "Nike running shoes for men black color"
            X = vectorizer.transform([test_text])
            prediction = model.predict(X)[0]
            probas = model.predict_proba(X)[0]
            max_proba = probas.max()
            
            print(f"   Texte: '{test_text}'")
            print(f"   ✅ Prédiction: {prediction}")
            print(f"   Confiance: {max_proba:.2%}")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    print("\n✅ Tests terminés!")

if __name__ == "__main__":
    test_model_loading()
