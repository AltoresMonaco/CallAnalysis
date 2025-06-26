#!/bin/bash

# Optimized Ollama startup script for H100 GPU
echo "=== Starting Optimized Ollama for H100 ==="

# Kill any existing Ollama processes
echo "Stopping existing Ollama processes..."
pkill -f ollama 2>/dev/null
sleep 2

# Start Ollama with optimized settings
echo "Starting Ollama with optimized settings..."
export OLLAMA_NUM_PARALLEL=4          # Allow 4 parallel requests
export OLLAMA_FLASH_ATTENTION=1       # Enable flash attention for speed
export OLLAMA_KEEP_ALIVE=-1           # Keep models cached indefinitely
export OLLAMA_MAX_LOADED_MODELS=1     # Keep only 1 model loaded
export OLLAMA_HOST=0.0.0.0:11434      # Bind to all interfaces

# Start Ollama in background
nohup ollama serve > /tmp/ollama_optimized.log 2>&1 &
OLLAMA_PID=$!

echo "Ollama started with PID: $OLLAMA_PID"
echo "Waiting for Ollama to be ready..."

# Wait for Ollama to start
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 1
done

# Check if Ollama started successfully
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Failed to start Ollama"
    exit 1
fi

# Pre-load the model
echo "🔥 Pre-loading model: lucifers/Polaris-4B-Preview.Q8_0"
curl -s -X POST http://localhost:11434/api/generate -d '{
    "model": "lucifers/Polaris-4B-Preview.Q8_0",
    "prompt": "System initialization",
    "stream": false,
    "keep_alive": -1
}' > /dev/null

# Check GPU memory usage
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "✅ Model loaded! GPU memory usage: ${GPU_MEMORY}MB"

echo "=== Ollama Optimization Complete ==="
echo "Settings:"
echo "  - Parallel requests: 4"
echo "  - Flash attention: Enabled"
echo "  - Keep alive: Indefinite"
echo "  - Model cached: Yes"
echo "  - GPU memory: ${GPU_MEMORY}MB"
echo ""
echo "To keep model warm continuously, run:"
echo "  ./keep_model_warm.sh &"
echo ""
echo "Ollama log: /tmp/ollama_optimized.log" 