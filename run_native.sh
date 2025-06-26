#!/bin/bash

# Function to detect cuDNN path
detect_cudnn_path() {
    echo "🔍 Detecting cuDNN installation..."
    
    # List of possible cuDNN locations (in order of preference)
    CUDNN_PATHS=(
        "/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib"
        "/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib"
        "/usr/local/lib/python3.9/dist-packages/nvidia/cudnn/lib"
        "/usr/local/cuda/lib64"
        "/usr/lib/x86_64-linux-gnu"
        "/opt/conda/lib"
        "/conda/lib"
    )
    
    for path in "${CUDNN_PATHS[@]}"; do
        if [ -d "$path" ] && [ -f "$path/libcudnn_ops.so.9" -o -f "$path/libcudnn_ops.so" ]; then
            echo "✅ Found cuDNN at: $path"
            export LD_LIBRARY_PATH="$path:$LD_LIBRARY_PATH"
            return 0
        fi
    done
    
    echo "⚠️  cuDNN not found in standard locations. Trying without explicit path..."
    return 1
}

# Activate virtual environment
source venv/bin/activate

# Detect and configure cuDNN
detect_cudnn_path

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
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""
echo "Web interface will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop the application"
echo ""

python app.py
