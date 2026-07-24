FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt requirements-dashboard.txt ./
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# Copy project
COPY . .

# Create output dirs
RUN mkdir -p outputs/shap data/vectorstore models/saved

EXPOSE 8000 8501
