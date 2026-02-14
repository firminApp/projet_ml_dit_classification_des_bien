"""
Script de test pour l'API de classification
"""

import time
import requests
from typing import Dict

# Configuration
API_BASE_URL = "http://localhost:8000"

# Couleurs pour l'affichage
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_test(name: str):
    """Affiche le nom du test."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Test: {name}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")


def print_success(message: str):
    """Affiche un message de succès."""
    print(f"{GREEN}✓ {message}{RESET}")


def print_error(message: str):
    """Affiche un message d'erreur."""
    print(f"{RED}✗ {message}{RESET}")


def print_info(message: str):
    """Affiche un message d'information."""
    print(f"{YELLOW}ℹ {message}{RESET}")


def test_health_check():
    """Test du health check."""
    print_test("Health Check")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Status: {response.status_code}")
            print_info(f"API Status: {data.get('status')}")
            print_info(f"Models Loaded: {data.get('models_loaded')}")
            print_info(f"Version: {data.get('version')}")
            
            if data.get('model_info'):
                model_info = data['model_info']
                print_info(f"Vectorizer: {model_info.get('vectorizer', {}).get('type')}")
                print_info(f"Classifier: {model_info.get('classifier', {}).get('type')}")
                print_info(f"Classes: {model_info.get('classifier', {}).get('n_classes')}")
            
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Impossible de se connecter à l'API")
        print_info("Assurez-vous que l'API est démarrée: python api_app.py")
        return False
    except Exception as e:
        print_error(f"Erreur: {e}")
        return False


def test_simple_predict():
    """Test de prédiction simple."""
    print_test("Prédiction Simple (Form Data)")
    
    try:
        data = {
            "text": "Nike running shoes for men black color breathable mesh"
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict/simple",
            data=data,
            timeout=10
        )
        elapsed = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print_success(f"Status: {response.status_code}")
            print_info(f"Catégorie: {result.get('category')}")
            print_info(f"Confiance: {result.get('confidence', 0):.2%}")
            print_info(f"Temps: {elapsed:.2f}ms")
            return True
        else:
            print_error(f"Status: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Erreur: {e}")
        return False


def test_predict():
    """Test de prédiction complète."""
    print_test("Prédiction Complète (JSON)")
    
    try:
        data = {
            "product_name": "Nike Air Zoom Pegasus",
            "description": "Premium running shoes for men with comfortable sole, black color, breathable mesh upper, perfect for jogging and fitness",
            "brand": "Nike"
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict?top_k=5",
            json=data,
            timeout=10
        )
        elapsed = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print_success(f"Status: {response.status_code}")
            print_info(f"Prédiction principale: {result.get('prediction')}")
            print_info(f"Confiance: {result.get('confidence', 0):.2%}")
            
            print(f"\n{YELLOW}Top 5 prédictions:{RESET}")
            for pred in result.get('top_k_predictions', [])[:5]:
                print(f"  {pred['rank']}. {pred['category'][:60]}... ({pred['confidence']:.2%})")
            
            metadata = result.get('metadata', {})
            print(f"\n{YELLOW}Métadonnées:{RESET}")
            print_info(f"Longueur texte: {metadata.get('text_length')} caractères")
            print_info(f"Nombre de classes: {metadata.get('n_classes')}")
            print_info(f"Temps traitement: {metadata.get('processing_time_ms', 0):.2f}ms")
            print_info(f"Temps total: {elapsed:.2f}ms")
            
            return True
        else:
            print_error(f"Status: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Erreur: {e}")
        return False


def test_batch_predict():
    """Test de prédiction par lot."""
    print_test("Prédiction par Lot")
    
    try:
        data = {
            "products": [
                {
                    "product_name": "Nike Running Shoes",
                    "description": "Comfortable running shoes",
                    "brand": "Nike"
                },
                {
                    "product_name": "Cotton Bedsheet",
                    "description": "King size cotton bedsheet with floral design, includes 2 pillow covers"
                },
                {
                    "description": "Stainless steel kitchen knife set, professional grade, 5 pieces"
                }
            ],
            "top_k": 3
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/batch-predict",
            json=data,
            timeout=30
        )
        elapsed = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print_success(f"Status: {response.status_code}")
            print_info(f"Produits traités: {result.get('total_processed')}")
            print_info(f"Temps total: {result.get('total_time_ms', 0):.2f}ms")
            print_info(f"Temps moyen par produit: {result.get('total_time_ms', 0) / max(result.get('total_processed', 1), 1):.2f}ms")
            
            print(f"\n{YELLOW}Prédictions:{RESET}")
            for i, pred in enumerate(result.get('predictions', []), 1):
                print(f"\n  Produit {i}:")
                print(f"    Catégorie: {pred.get('prediction')[:60]}...")
                print(f"    Confiance: {pred.get('confidence', 0):.2%}")
            
            return True
        else:
            print_error(f"Status: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Erreur: {e}")
        return False


def test_models_info():
    """Test de récupération des infos modèles."""
    print_test("Informations sur les Modèles")
    
    try:
        response = requests.get(f"{API_BASE_URL}/models/info", timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print_success(f"Status: {response.status_code}")
            
            models = result.get('models', {})
            
            print(f"\n{YELLOW}Vectorizer:{RESET}")
            vec_info = models.get('vectorizer', {})
            print_info(f"Type: {vec_info.get('type')}")
            print_info(f"Features: {vec_info.get('max_features')}")
            print_info(f"N-grams: {vec_info.get('ngram_range')}")
            
            print(f"\n{YELLOW}Classifier:{RESET}")
            clf_info = models.get('classifier', {})
            print_info(f"Type: {clf_info.get('type')}")
            print_info(f"Nombre de classes: {clf_info.get('n_classes')}")
            print_info(f"Exemples de classes: {clf_info.get('classes_sample')}")
            
            print(f"\n{YELLOW}Chemins:{RESET}")
            paths = result.get('paths', {})
            print_info(f"Vectorizer: {paths.get('vectorizer')}")
            print_info(f"Classifier: {paths.get('classifier')}")
            
            return True
        else:
            print_error(f"Status: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Erreur: {e}")
        return False


def test_error_handling():
    """Test de la gestion d'erreurs."""
    print_test("Gestion d'Erreurs")
    
    # Test 1: Requête vide
    print(f"\n{YELLOW}Test 1: Description vide{RESET}")
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"description": ""},
            timeout=5
        )
        
        if response.status_code == 400:
            print_success("Erreur 400 correctement retournée")
        else:
            print_error(f"Status inattendu: {response.status_code}")
    except Exception as e:
        print_error(f"Erreur: {e}")
    
    # Test 2: JSON invalide
    print(f"\n{YELLOW}Test 2: JSON invalide{RESET}")
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        if response.status_code == 422:
            print_success("Erreur 422 correctement retournée")
        else:
            print_error(f"Status inattendu: {response.status_code}")
    except Exception as e:
        print_error(f"Erreur: {e}")
    
    return True


def run_all_tests():
    """Exécute tous les tests."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}🧪 Tests de l'API CLF04{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{YELLOW}URL de l'API: {API_BASE_URL}{RESET}")
    
    tests = [
        ("Health Check", test_health_check),
        ("Prédiction Simple", test_simple_predict),
        ("Prédiction Complète", test_predict),
        ("Prédiction par Lot", test_batch_predict),
        ("Informations Modèles", test_models_info),
        ("Gestion d'Erreurs", test_error_handling)
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"Erreur lors du test {test_name}: {e}")
            results[test_name] = False
        
        time.sleep(0.5)  # Pause entre les tests
    
    total_time = time.time() - start_time
    
    # Résumé
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}📊 Résumé des Tests{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{GREEN}✓ PASS{RESET}" if result else f"{RED}✗ FAIL{RESET}"
        print(f"{status} - {test_name}")
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{YELLOW}Tests réussis: {passed}/{total}{RESET}")
    print(f"{YELLOW}Temps total: {total_time:.2f}s{RESET}")
    
    if passed == total:
        print(f"{GREEN}✓ Tous les tests ont réussi!{RESET}")
        return 0
    else:
        print(f"{RED}✗ Certains tests ont échoué{RESET}")
        return 1


if __name__ == "__main__":
    import sys
    
    print(f"\n{YELLOW}💡 Assurez-vous que l'API est démarrée:{RESET}")
    print(f"{YELLOW}   cd api && python api_app.py{RESET}\n")
    
    exit_code = run_all_tests()
    sys.exit(exit_code)
