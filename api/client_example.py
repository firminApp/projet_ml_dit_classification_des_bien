"""
Client Python pour l'API de classification CLF04

Exemple d'utilisation de l'API avec requests
"""

import requests
from typing import Dict, List, Optional


class CLF04Client:
    """Client Python pour l'API de classification."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialise le client API.
        
        Args:
            base_url: URL de base de l'API
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict:
        """
        Vérifie l'état de santé de l'API.
        
        Returns:
            Dictionnaire avec le statut de l'API
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict(
        self,
        description: str,
        product_name: Optional[str] = None,
        brand: Optional[str] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Prédit la catégorie d'un produit.
        
        Args:
            description: Description du produit en anglais
            product_name: Nom du produit (optionnel)
            brand: Marque du produit (optionnel)
            top_k: Nombre de prédictions à retourner
        
        Returns:
            Dictionnaire avec les prédictions
        """
        data = {"description": description}
        
        if product_name:
            data["product_name"] = product_name
        if brand:
            data["brand"] = brand
        
        response = requests.post(
            f"{self.base_url}/predict",
            json=data,
            params={"top_k": top_k}
        )
        response.raise_for_status()
        return response.json()
    
    def predict_simple(self, text: str) -> Dict:
        """
        Prédiction simple avec un seul champ texte.
        
        Args:
            text: Texte à classifier
        
        Returns:
            Dictionnaire avec la prédiction principale
        """
        response = requests.post(
            f"{self.base_url}/predict/simple",
            data={"text": text}
        )
        response.raise_for_status()
        return response.json()
    
    def batch_predict(
        self,
        products: List[Dict[str, str]],
        top_k: int = 5
    ) -> Dict:
        """
        Prédit les catégories pour plusieurs produits.
        
        Args:
            products: Liste de produits avec description, nom, marque
            top_k: Nombre de prédictions par produit
        
        Returns:
            Dictionnaire avec toutes les prédictions
        """
        data = {
            "products": products,
            "top_k": top_k
        }
        
        response = requests.post(
            f"{self.base_url}/batch-predict",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def get_models_info(self) -> Dict:
        """
        Récupère les informations sur les modèles.
        
        Returns:
            Dictionnaire avec les infos des modèles
        """
        response = requests.get(f"{self.base_url}/models/info")
        response.raise_for_status()
        return response.json()


# ===== Exemples d'utilisation =====

def example_simple():
    """Exemple simple de prédiction."""
    print("=" * 60)
    print("Exemple 1: Prédiction Simple")
    print("=" * 60)
    
    client = CLF04Client()
    
    # Vérifier que l'API est disponible
    health = client.health_check()
    print(f"✓ API Status: {health['status']}")
    print()
    
    # Prédiction simple
    result = client.predict_simple(
        "Nike running shoes for men black color breathable mesh"
    )
    
    print(f"Catégorie prédite: {result['category']}")
    print(f"Confiance: {result['confidence']:.2%}")
    print()


def example_complete():
    """Exemple complet de prédiction."""
    print("=" * 60)
    print("Exemple 2: Prédiction Complète")
    print("=" * 60)
    
    client = CLF04Client()
    
    result = client.predict(
        product_name="Nike Air Zoom Pegasus",
        description="Premium running shoes for men with comfortable sole, black color, breathable mesh upper, perfect for jogging and fitness",
        brand="Nike",
        top_k=5
    )
    
    print(f"Prédiction principale: {result['prediction']}")
    print(f"Confiance: {result['confidence']:.2%}")
    print()
    
    print("Top 5 prédictions:")
    for pred in result['top_k_predictions']:
        cat = pred['category'][:60] + '...' if len(pred['category']) > 60 else pred['category']
        print(f"  {pred['rank']}. {cat} ({pred['confidence']:.2%})")
    print()
    
    metadata = result['metadata']
    print(f"Métadonnées:")
    print(f"  - Longueur texte: {metadata['text_length']} caractères")
    print(f"  - Nombre de classes: {metadata['n_classes']}")
    print(f"  - Temps traitement: {metadata['processing_time_ms']:.2f}ms")
    print()


def example_batch():
    """Exemple de prédiction par lot."""
    print("=" * 60)
    print("Exemple 3: Prédiction par Lot")
    print("=" * 60)
    
    client = CLF04Client()
    
    products = [
        {
            "product_name": "Nike Running Shoes",
            "description": "Comfortable running shoes for men",
            "brand": "Nike"
        },
        {
            "product_name": "Cotton Bedsheet",
            "description": "King size cotton bedsheet with floral design, includes 2 pillow covers"
        },
        {
            "description": "Stainless steel kitchen knife set, professional grade, 5 pieces"
        }
    ]
    
    result = client.batch_predict(products, top_k=3)
    
    print(f"Produits traités: {result['total_processed']}")
    print(f"Temps total: {result['total_time_ms']:.2f}ms")
    print()
    
    for i, pred in enumerate(result['predictions'], 1):
        print(f"Produit {i}:")
        cat = pred['prediction'][:60] + '...' if len(pred['prediction']) > 60 else pred['prediction']
        print(f"  Catégorie: {cat}")
        print(f"  Confiance: {pred['confidence']:.2%}")
        print()


def example_models_info():
    """Exemple d'informations sur les modèles."""
    print("=" * 60)
    print("Exemple 4: Informations sur les Modèles")
    print("=" * 60)
    
    client = CLF04Client()
    
    info = client.get_models_info()
    
    models = info['models']
    
    print("Vectorizer:")
    vec = models['vectorizer']
    print(f"  Type: {vec['type']}")
    print(f"  Features: {vec['max_features']}")
    print(f"  N-grams: {vec['ngram_range']}")
    print()
    
    print("Classifier:")
    clf = models['classifier']
    print(f"  Type: {clf['type']}")
    print(f"  Nombre de classes: {clf['n_classes']}")
    print(f"  Exemples: {clf['classes_sample']}")
    print()


def example_error_handling():
    """Exemple de gestion d'erreurs."""
    print("=" * 60)
    print("Exemple 5: Gestion d'Erreurs")
    print("=" * 60)
    
    client = CLF04Client()
    
    try:
        # Tenter une prédiction avec description vide
        result = client.predict(description="")
    except requests.exceptions.HTTPError as e:
        print(f"✓ Erreur HTTP capturée: {e.response.status_code}")
        error_data = e.response.json()
        print(f"  Message: {error_data.get('error')}")
        print()
    
    try:
        # Tenter de se connecter à une URL invalide
        client_invalid = CLF04Client("http://localhost:9999")
        client_invalid.health_check()
    except requests.exceptions.ConnectionError:
        print("✓ Erreur de connexion capturée")
        print("  L'API n'est pas accessible")
        print()


if __name__ == "__main__":
    print("\n🐍 Client Python - API CLF04\n")
    
    try:
        example_simple()
        example_complete()
        example_batch()
        example_models_info()
        example_error_handling()
        
        print("=" * 60)
        print("✅ Tous les exemples ont été exécutés avec succès!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Erreur: Impossible de se connecter à l'API")
        print("   Assurez-vous que l'API est démarrée:")
        print("   cd api && python api_app.py")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
