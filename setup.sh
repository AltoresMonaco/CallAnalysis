#!/bin/bash

# Audio Transcript App - Native Setup Script
# This script sets up the environment to run the app natively (without Docker)
#
# If you pasted this file, make it executable first:
#   chmod +x setup.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "Setting up Audio Transcript App for native execution..."

# Self-check: Ensure this script is executable
if [ ! -x "$0" ]; then
    print_warning "This script is not executable. Making it executable..."
    chmod +x "$0"
    print_success "Script is now executable!"
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Found Python $PYTHON_VERSION"

# Create virtual environment
print_status "Creating Python virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Removing and recreating..."
    rm -rf venv
fi

python3 -m venv venv
print_success "Virtual environment created!"

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Detect GPU availability and choose appropriate requirements
print_status "Detecting hardware capabilities..."

# Check for NVIDIA GPU
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi > /dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        if [ $GPU_COUNT -gt 0 ]; then
            GPU_AVAILABLE=true
            print_success "NVIDIA GPU detected: $GPU_COUNT device(s)"
            nvidia-smi --query-gpu=name --format=csv,noheader | head -3
        fi
    fi
fi

# Choose requirements file based on GPU availability
if [ "$GPU_AVAILABLE" = true ] && [ -f "requirements-gpu.txt" ]; then
    REQUIREMENTS_FILE="requirements-gpu.txt"
    export USE_GPU=true
    print_status "Using GPU-optimized requirements (faster-whisper, PyTorch CUDA)"
else
    REQUIREMENTS_FILE="requirements.txt"
    export USE_GPU=false
    print_status "Using CPU-only requirements (openai-whisper)"
fi

# Install requirements
print_status "Installing Python dependencies from $REQUIREMENTS_FILE..."
pip install -r $REQUIREMENTS_FILE

print_success "Dependencies installed with $([ "$GPU_AVAILABLE" = true ] && echo "GPU" || echo "CPU") support!"

# Check if Ollama is running
print_status "Checking Ollama connection..."
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

if curl -s "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
    print_success "Ollama is accessible at $OLLAMA_URL"
    
    # Check if llama3 model is available
    if curl -s "$OLLAMA_URL/api/tags" | grep -q "llama3"; then
        print_success "llama3 model is available"
    else
        print_warning "llama3 model not found. You may need to pull it:"
        echo "  ollama pull llama3"
    fi
else
    print_warning "Cannot connect to Ollama at $OLLAMA_URL"
    print_warning "Make sure Ollama is running: ollama serve"
fi

# Create directories
print_status "Creating necessary directories..."
mkdir -p saved_results
mkdir -p settings
mkdir -p prompts

print_success "Setup completed!"

# Create a simple run script with detected GPU settings
print_status "Creating run script..."
cat > run_native.sh << EOF
#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set environment variables (auto-detected during setup)
export OLLAMA_URL="\${OLLAMA_URL:-http://localhost:11434}"
export OLLAMA_MODEL="\${OLLAMA_MODEL:-llama3}"
export USE_GPU="\${USE_GPU:-$USE_GPU}"
export MAX_WORKERS="\${MAX_WORKERS:-$([ "$GPU_AVAILABLE" = true ] && echo "20" || echo "4")}"

# Show hardware configuration
echo "🚀 Starting Audio Transcript App..."
echo "Hardware: $([ "$GPU_AVAILABLE" = true ] && echo "GPU-accelerated" || echo "CPU-only")"
echo "Whisper: $([ "$GPU_AVAILABLE" = true ] && echo "faster-whisper (GPU)" || echo "openai-whisper (CPU)")"
echo "Max Workers: \$MAX_WORKERS"
echo ""
echo "Web interface will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop the application"
echo ""

python app.py
EOF

chmod +x run_native.sh
print_success "Run script created: run_native.sh"

# Create environment file for easy customization
cat > .env.example << 'EOF'
# Audio Transcript App Environment Variables
# Copy this file to .env and customize as needed

# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Performance Settings
USE_GPU=false
MAX_WORKERS=4

# Optional: Memory limit (GB)
MEMORY_LIMIT_GB=8
EOF

print_success "Example environment file created: .env.example"

echo ""
echo "🎉 Setup Complete!"
echo ""
echo "Hardware Configuration:"
echo "  • Mode: $([ "$GPU_AVAILABLE" = true ] && echo "GPU-accelerated 🚀" || echo "CPU-only 💻")"
echo "  • Whisper: $([ "$GPU_AVAILABLE" = true ] && echo "faster-whisper (GPU)" || echo "openai-whisper (CPU)")"
echo "  • Max Workers: $([ "$GPU_AVAILABLE" = true ] && echo "20" || echo "4")"
echo ""
echo "To run the application:"
echo "  ./run_native.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "The app will be available at: http://localhost:8000"
echo ""
echo "Available interfaces:"
echo "  • Main App: http://localhost:8000"
echo "  • IDE Tool: http://localhost:8000/ide"
echo "  • Browser: http://localhost:8000/browser"
echo "  • API Docs: http://localhost:8000/docs"
echo ""
echo "Make sure Ollama is running before starting the app:"
echo "  ollama serve" 