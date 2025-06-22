#!/bin/bash

# Audio Transcription API - Client Setup
# This script helps clients quickly start the API with their existing Ollama

set -e

echo "🎵 Audio Transcription API - Client Setup"
echo "========================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Ollama is accessible
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
echo "🔍 Checking Ollama connection at $OLLAMA_URL..."

if curl -s "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
    echo "✅ Ollama is accessible"
    
    # Check if model exists
    MODEL="${OLLAMA_MODEL:-llama3}"
    if curl -s "$OLLAMA_URL/api/tags" | grep -q "$MODEL"; then
        echo "✅ Model '$MODEL' is available"
    else
        echo "⚠️  Model '$MODEL' not found. Please run: ollama pull $MODEL"
        exit 1
    fi
else
    echo "❌ Cannot connect to Ollama at $OLLAMA_URL"
    echo "Please make sure Ollama is running:"
    echo "  ollama serve"
    exit 1
fi

# Create settings volume
echo "📁 Creating settings volume..."
docker volume create audio-transcription-settings 2>/dev/null || true

# Pull and start the API
echo "🚀 Starting Audio Transcription API..."
docker run -d \
  --name audio-transcription-api \
  -p 8000:8000 \
  -e OLLAMA_URL="$OLLAMA_URL/api/generate" \
  -e OLLAMA_MODEL="$MODEL" \
  -v /tmp/audio_processing:/tmp/audio_processing \
  -v audio-transcription-settings:/app/settings \
  --restart unless-stopped \
  YOUR_DOCKERHUB_USERNAME/audio-transcription-api

echo "✅ API started successfully!"
echo ""
echo "🌐 Web Interface: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "💚 Health Check: http://localhost:8000/health"
echo ""
echo "💡 You can now configure Ollama settings through the web interface!"
echo "   Click the gear icon (⚙️) in the top-right corner."
echo ""
echo "To stop: docker stop audio-transcription-api"
echo "To remove: docker rm audio-transcription-api"
echo "Settings are preserved in: audio-transcription-settings volume" 