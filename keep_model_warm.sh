#!/bin/bash

# Script to keep Ollama model cached in GPU memory
MODEL_NAME="lucifers/Polaris-4B-Preview.Q8_0"

echo "=== Keeping Model Warm Script ==="
echo "Model: $MODEL_NAME"
echo "Starting at: $(date)"

# Function to check if model is loaded
check_model_loaded() {
    gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$gpu_memory" -gt 10000 ]; then
        echo "✅ Model appears to be loaded (GPU memory: ${gpu_memory}MB)"
        return 0
    else
        echo "❌ Model not loaded (GPU memory: ${gpu_memory}MB)"
        return 1
    fi
}

# Function to warm up model
warm_model() {
    echo "🔥 Warming up model..."
    curl -s -X POST http://localhost:11434/api/generate -d "{
        \"model\": \"$MODEL_NAME\",
        \"prompt\": \"System ready\",
        \"stream\": false,
        \"keep_alive\": -1
    }" > /dev/null
    
    if [ $? -eq 0 ]; then
        echo "✅ Model warmed up successfully"
    else
        echo "❌ Failed to warm up model"
        return 1
    fi
}

# Function to keep model alive
keep_alive() {
    echo "💓 Sending keep-alive ping..."
    curl -s -X POST http://localhost:11434/api/generate -d "{
        \"model\": \"$MODEL_NAME\",
        \"prompt\": \"\",
        \"keep_alive\": -1
    }" > /dev/null
}

# Main execution
echo "Checking if Ollama is running..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama is not running! Please start Ollama first."
    exit 1
fi

echo "✅ Ollama is running"

# Initial model warm-up
if ! check_model_loaded; then
    warm_model
    sleep 3
    check_model_loaded
fi

# Keep model alive with periodic pings
echo "🔄 Starting keep-alive loop (every 5 minutes)..."
echo "Press Ctrl+C to stop"

while true; do
    if ! check_model_loaded; then
        echo "🔥 Model unloaded, reloading..."
        warm_model
    else
        keep_alive
    fi
    
    echo "💤 Waiting 5 minutes... ($(date))"
    sleep 300  # 5 minutes
done 