FROM python:3.10-slim

# Preventing Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installing system dependencies (single layer + cleanup)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better caching)
COPY backend/flask_app/requirements.txt /app/requirements.txt

# Install Python deps WITHOUT cache
RUN pip install --no-cache-dir -r requirements.txt

# Download nltk data (still needed)
RUN python -m nltk.downloader stopwords wordnet

# Copy application code LAST
COPY backend/flask_app/ /app/

EXPOSE 5000

CMD ["python", "app.py"]
