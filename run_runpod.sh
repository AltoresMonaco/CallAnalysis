#!/bin/bash

echo "🚀 Starting Monaco Telecom Audio Analysis on RunPod H100"

# Configuration GPU
export USE_GPU=true
export CUDA_VISIBLE_DEVICES=0
export SKIP_DIARIZATION=false
export MAX_WORKERS=10
export OLLAMA_URL=http://localhost:11434/api/generate
export OLLAMA_MODEL=llama3.1:latest

# Vérifications
echo "📊 GPU Check:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

echo "🔧 Python Environment:"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not installed yet"

# Installation dépendances si nécessaire
if [ ! -f "requirements_installed.flag" ]; then
    echo "📦 Installing requirements..."
    pip install -r requirements-gpu.txt
    touch requirements_installed.flag
    echo "✅ Requirements installed"
else
    echo "✅ Requirements already installed"
fi

# Vérification post-installation
echo "🧪 Post-install verification:"
python -c "
import torch
print(f'PyTorch CUDA: {torch.cuda.is_available()}')
try:
    from faster_whisper import WhisperModel
    print('✅ faster-whisper available')
except ImportError as e:
    print(f'❌ faster-whisper import failed: {e}')
try:
    from pyannote.audio import Pipeline
    print('✅ pyannote.audio available')
except ImportError as e:
    print(f'❌ pyannote.audio import failed: {e}')
"

# Démarrage Ollama en arrière-plan
if ! pgrep -f "ollama" > /dev/null; then
    echo "🤖 Starting Ollama..."
    if command -v ollama &> /dev/null; then
        ollama serve &
        sleep 5
        ollama pull llama3.1:latest 2>/dev/null || echo "⚠️  Ollama model pull failed, continuing without it"
    else
        echo "⚠️  Ollama not found, continuing without it"
    fi
else
    echo "✅ Ollama already running"
fi

# Démarrage application
echo "🎯 Starting FastAPI application..."
echo "Environment variables:"
echo "- USE_GPU: $USE_GPU"
echo "- SKIP_DIARIZATION: $SKIP_DIARIZATION"
echo "- MAX_WORKERS: $MAX_WORKERS"

echo "🔧 Diagnostic et résolution cuDNN automatique..."
python fix_cudnn_gpu.py

if [ $? -eq 0 ]; then
    echo "✅ cuDNN résolu avec succès!"
    echo "🚀 Démarrage application Monaco Telecom sur RunPod H100..."
    python app.py
else
    echo "❌ Échec résolution cuDNN - démarrage en mode CPU de secours"
    export USE_GPU=false
    python app.py
fi 