FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'API
COPY api_app.py .

# Copier les modèles (à ajuster selon votre structure)
COPY ../models /app/models

# Exposer le port
EXPOSE 8000

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Lancer l'API
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000"]
