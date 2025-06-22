# 🚀 Docker Hub Deployment Guide

This guide shows how to build and deploy the Audio Transcription API to Docker Hub for easy client distribution.

## 📦 Building and Publishing

### 1. Build the Docker Image

```bash
# Build the image
docker build -t audio-transcription-api .

# Tag for Docker Hub (replace with your username)
docker tag audio-transcription-api YOUR_DOCKERHUB_USERNAME/audio-transcription-api:latest
docker tag audio-transcription-api YOUR_DOCKERHUB_USERNAME/audio-transcription-api:v1.0
```

### 2. Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Push the images
docker push YOUR_DOCKERHUB_USERNAME/audio-transcription-api:latest
docker push YOUR_DOCKERHUB_USERNAME/audio-transcription-api:v1.0
```

### 3. Update Client Files

Before distributing, update the placeholder in these files:

**docker-compose-client.yml:**
```yaml
image: YOUR_DOCKERHUB_USERNAME/audio-transcription-api
```

**client-setup.sh:**
```bash
YOUR_DOCKERHUB_USERNAME/audio-transcription-api
```

Replace `YOUR_DOCKERHUB_USERNAME` with your actual Docker Hub username.

## 📋 Client Distribution

### Option 1: Docker Compose (Recommended)

Share these files with clients:
- `docker-compose-client.yml`
- Instructions to run: `docker-compose -f docker-compose-client.yml up -d`

### Option 2: Simple Script

Share the `client-setup.sh` script:
```bash
chmod +x client-setup.sh
./client-setup.sh
```

### Option 3: Manual Docker Run

Clients can run directly:
```bash
docker run -d \
  --name audio-transcription-api \
  -p 8000:8000 \
  -e OLLAMA_URL=http://host.docker.internal:11434/api/generate \
  -e OLLAMA_MODEL=llama3 \
  -v /tmp/audio_processing:/tmp/audio_processing \
  -v audio-transcription-settings:/app/settings \
  --restart unless-stopped \
  YOUR_DOCKERHUB_USERNAME/audio-transcription-api
```

## ✅ Client Requirements

Clients need:
1. **Docker installed**
2. **Ollama running with a model** (e.g., `ollama pull llama3`)

## 🎯 Key Features for Clients

- **🌐 Web Interface**: http://localhost:8000
- **⚙️ Settings Management**: Click the gear icon to configure Ollama URL/model
- **💾 Persistent Settings**: Settings are saved between container restarts
- **🔌 Connection Testing**: Built-in Ollama connection testing
- **📱 Mobile Responsive**: Works on all devices

## 🔧 Configuration

Clients can configure the API in two ways:

### 1. Environment Variables (Initial Setup)
```bash
OLLAMA_URL=http://192.168.1.100:11434 \
OLLAMA_MODEL=mistral \
docker-compose -f docker-compose-client.yml up -d
```

### 2. Web Interface (Runtime)
- Open http://localhost:8000
- Click the gear icon (⚙️)
- Enter Ollama URL and model
- Test connection and save

## 🚨 Troubleshooting

**Cannot connect to Ollama:**
- Verify Ollama is running: `ollama serve`
- Check the URL in settings
- Ensure the model is pulled: `ollama pull MODEL_NAME`

**Settings not persisting:**
- Settings are stored in Docker volume: `audio-transcription-settings`
- Don't remove this volume when updating containers

**Port conflicts:**
- Change port mapping: `-p 8080:8000` (access via http://localhost:8080)

## 📊 Version Management

Tag versions for better client management:
```bash
# Create version tags
docker tag YOUR_DOCKERHUB_USERNAME/audio-transcription-api:latest \
     YOUR_DOCKERHUB_USERNAME/audio-transcription-api:v1.1

# Push specific version
docker push YOUR_DOCKERHUB_USERNAME/audio-transcription-api:v1.1
```

Clients can specify versions:
```yaml
image: YOUR_DOCKERHUB_USERNAME/audio-transcription-api:v1.1
``` 