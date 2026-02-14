#!/bin/bash

# Exemples de requêtes curl pour tester l'API
# CLF04 - Classification de biens de consommation

API_URL="http://localhost:8000"

echo "🧪 Exemples de requêtes API - CLF04"
echo "===================================="
echo ""

# Vérifier si l'API est accessible
echo "1️⃣  Health Check"
echo "----------------"
echo "curl $API_URL/health"
echo ""
curl -s $API_URL/health | python -m json.tool
echo ""
echo ""

# Prédiction simple avec form data
echo "2️⃣  Prédiction Simple (Form Data)"
echo "--------------------------------"
echo 'curl -X POST "$API_URL/predict/simple" -F "text=Nike running shoes for men"'
echo ""
curl -s -X POST "$API_URL/predict/simple" \
  -F "text=Nike running shoes for men black color breathable mesh" | python -m json.tool
echo ""
echo ""

# Prédiction complète avec JSON
echo "3️⃣  Prédiction Complète (JSON)"
echo "-----------------------------"
cat << 'EOF'
curl -X POST "$API_URL/predict?top_k=3" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Nike Air Zoom Pegasus",
    "description": "Premium running shoes for men with comfortable sole",
    "brand": "Nike"
  }'
EOF
echo ""
curl -s -X POST "$API_URL/predict?top_k=3" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Nike Air Zoom Pegasus",
    "description": "Premium running shoes for men with comfortable sole, black color, breathable mesh upper, perfect for jogging and fitness",
    "brand": "Nike"
  }' | python -m json.tool
echo ""
echo ""

# Prédiction d'un bedsheet
echo "4️⃣  Exemple: Bedsheet (Literie)"
echo "------------------------------"
cat << 'EOF'
curl -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Cotton Bedsheet",
    "description": "Premium quality cotton bedsheet with floral design, king size, includes 2 pillow covers, machine washable",
    "brand": "Elegance"
  }'
EOF
echo ""
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Cotton Bedsheet",
    "description": "Premium quality cotton bedsheet with floral design, king size, includes 2 pillow covers, machine washable",
    "brand": "Elegance"
  }' | python -m json.tool
echo ""
echo ""

# Prédiction par lot
echo "5️⃣  Prédiction par Lot (3 produits)"
echo "----------------------------------"
cat << 'EOF'
curl -X POST "$API_URL/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "products": [
      {"description": "Nike running shoes"},
      {"description": "Cotton bedsheet king size"},
      {"description": "Kitchen knife set professional grade"}
    ],
    "top_k": 2
  }'
EOF
echo ""
curl -s -X POST "$API_URL/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "products": [
      {
        "product_name": "Nike Shoes",
        "description": "Running shoes for men",
        "brand": "Nike"
      },
      {
        "product_name": "Cotton Bedsheet",
        "description": "King size cotton bedsheet with floral design"
      },
      {
        "description": "Stainless steel kitchen knife set, professional grade, 5 pieces"
      }
    ],
    "top_k": 2
  }' | python -m json.tool
echo ""
echo ""

# Informations sur les modèles
echo "6️⃣  Informations sur les Modèles"
echo "--------------------------------"
echo "curl $API_URL/models/info"
echo ""
curl -s $API_URL/models/info | python -m json.tool
echo ""
echo ""

echo "✅ Tests terminés!"
echo ""
echo "📚 Pour plus d'informations:"
echo "   - Documentation Swagger: $API_URL/docs"
echo "   - Documentation ReDoc: $API_URL/redoc"
