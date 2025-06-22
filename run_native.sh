#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set environment variables (auto-detected during setup)
export OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3}"
export USE_GPU="${USE_GPU:-true}"
export MAX_WORKERS="${MAX_WORKERS:-20}"

# Show hardware configuration
echo "🚀 Starting Audio Transcript App..."
echo "Hardware: GPU-accelerated"
echo "Whisper: faster-whisper (GPU)"
echo "Max Workers: $MAX_WORKERS"
echo ""
echo "Web interface will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop the application"
echo ""

python app.py
