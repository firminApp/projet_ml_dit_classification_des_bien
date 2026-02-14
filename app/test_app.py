#!/usr/bin/env python3
"""
Script de test pour vérifier que l'application Streamlit est prête
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    print("🔍 Vérification des dépendances...")
    
    missing = []
    required = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy'
    }
    
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Packages manquants: {', '.join(missing)}")
        print(f"Installez-les avec: pip install {' '.join(missing)}")
        return False
    
    print("✅ Toutes les dépendances sont installées!\n")
    return True

def check_models():
    """Vérifie la présence des modèles"""
    print("🔍 Vérification des modèles...")
    
    project_dir = Path(__file__).parent.parent
    model_dir = project_dir / "models"
    
    # Modèles optimisés
    optimized_files = [
        "optimized_model.pkl",
        "optimized_vectorizer.pkl",
        "optimized_scaler.pkl",
        "optimized_brand_encoder.pkl",
        "optimized_model_metadata.json"
    ]
    
    optimized_exists = all((model_dir / f).exists() for f in optimized_files)
    
    # Modèles baseline
    baseline_files = [
        "tfidf_vectorizer.pkl",
        "logistic_regression_model.pkl"
    ]
    
    baseline_exists = all((model_dir / f).exists() for f in baseline_files)
    
    if optimized_exists:
        print("  ✅ Modèles optimisés trouvés (Accuracy: ~69%)")
        for f in optimized_files:
            size = (model_dir / f).stat().st_size / 1024
            print(f"      └─ {f} ({size:.1f} KB)")
        return "optimized"
    elif baseline_exists:
        print("  ⚠️  Modèles baseline trouvés (Accuracy: ~21%)")
        for f in baseline_files:
            if (model_dir / f).exists():
                size = (model_dir / f).stat().st_size / 1024
                print(f"      └─ {f} ({size:.1f} KB)")
        print("\n  💡 Pour créer les modèles optimisés:")
        print("     1. Exécutez le notebook (cellules 88-100)")
        print("     2. Ou lancez: python train_best_model.py\n")
        return "baseline"
    else:
        print("  ❌ Aucun modèle trouvé!")
        print("\n  ⚠️  L'application ne pourra pas fonctionner sans modèles.")
        print("  📝 Pour créer les modèles:")
        print("     1. Option A (recommandé): Ouvrez classification_biens.ipynb")
        print("        - Exécutez d'abord la cellule 88 (filtrage)")
        print("        - Puis exécutez les cellules 90-100")
        print("     2. Option B: Lancez python train_best_model.py\n")
        return None

def check_data():
    """Vérifie la présence des données"""
    print("🔍 Vérification des données...")
    
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"
    
    required_files = [
        "flipkart_com-ecommerce_sample_1050.csv"
    ]
    
    all_exist = True
    for f in required_files:
        file_path = data_dir / f
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {f} ({size:.1f} MB)")
        else:
            print(f"  ❌ {f}")
            all_exist = False
    
    if all_exist:
        print("✅ Toutes les données sont présentes!\n")
    else:
        print("⚠️  Certains fichiers de données sont manquants\n")
    
    return all_exist

def test_model_loading():
    """Teste le chargement des modèles"""
    print("🔍 Test de chargement des modèles...")
    
    try:
        import pickle
        import json
        project_dir = Path(__file__).parent.parent
        model_dir = project_dir / "models"
        
        # Tester le chargement du modèle optimisé si disponible
        optimized_model_path = model_dir / "optimized_model.pkl"
        if optimized_model_path.exists():
            with open(optimized_model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"  ✅ Modèle chargé: {type(model).__name__}")
            print(f"  ✅ Nombre de classes: {len(model.classes_)}")
            print(f"  ✅ Premières classes: {model.classes_[:3]}")
            
            # Charger metadata
            metadata_path = model_dir / "optimized_model_metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"  ✅ Metadata chargé:")
            print(f"      - Test accuracy: {metadata['test_accuracy']:.1%}")
            print(f"      - Test F1: {metadata['test_f1_macro']:.1%}")
            print(f"      - TF-IDF features: {metadata['tfidf_n_features']}")
            print("\n✅ Modèles fonctionnels!\n")
            return True
        else:
            print("  ⚠️  Modèles optimisés non disponibles (test skippé)\n")
            return True
            
    except Exception as e:
        print(f"  ❌ Erreur lors du chargement: {e}\n")
        return False

def main():
    """Fonction principale"""
    print("="*70)
    print("🧪 TEST DE L'APPLICATION STREAMLIT")
    print("="*70)
    print()
    
    # Vérifications
    deps_ok = check_dependencies()
    model_status = check_models()
    data_ok = check_data()
    model_loading_ok = test_model_loading() if model_status else False
    
    # Résumé
    print("="*70)
    print("📋 RÉSUMÉ")
    print("="*70)
    
    if deps_ok and model_status and data_ok:
        print("✅ L'application est prête à être lancée!")
        print()
        print("🚀 Pour démarrer:")
        print("   streamlit run app/streamlit_app.py")
        print()
        
        if model_status == "optimized":
            print("💡 Vous utilisez les modèles optimisés (69.4% accuracy)")
        else:
            print("💡 Vous utilisez les modèles baseline (21% accuracy)")
            print("   Créez les modèles optimisés pour de meilleures performances!")
        
        return 0
    else:
        print("❌ Certains problèmes doivent être résolus avant de lancer l'application.")
        if not deps_ok:
            print("   - Installez les dépendances manquantes")
        if not model_status:
            print("   - Créez les modèles (notebook ou script)")
        if not data_ok:
            print("   - Vérifiez la présence des fichiers de données")
        return 1

if __name__ == "__main__":
    sys.exit(main())
