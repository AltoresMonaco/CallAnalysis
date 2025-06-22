#!/bin/bash

# Audio Transcription API Management Script
# This script helps manage the containerized audio transcription service

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

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
}

# Check external Ollama connection
check_ollama() {
    local ollama_url="${OLLAMA_URL:-http://localhost:11434}"
    print_status "Checking external Ollama connection at $ollama_url..."
    
    if curl -s "$ollama_url/api/tags" > /dev/null 2>&1; then
        print_success "External Ollama is accessible"
        
        # Check if the model is available
        local model="${OLLAMA_MODEL:-llama3}"
        if curl -s "$ollama_url/api/tags" | grep -q "$model"; then
            print_success "Model '$model' is available"
        else
            print_warning "Model '$model' not found. You may need to pull it:"
            echo "  ollama pull $model"
        fi
    else
        print_warning "Cannot connect to external Ollama at $ollama_url"
        echo "Make sure Ollama is running and accessible."
        echo "You can set a custom URL with: export OLLAMA_URL=http://your-ollama-host:11434"
    fi
}

# Build the application
build() {
    print_status "Building audio transcription API..."
    check_docker
    docker-compose build
    print_success "Build completed!"
}

# Start the services
up() {
    print_status "Starting audio transcription API..."
    check_docker
    check_ollama
    docker-compose up -d
    print_success "Services started!"
    print_status "Web interface: http://localhost:8000"
    print_status "API docs: http://localhost:8000/docs"
    print_status "Health check: http://localhost:8000/health"
}

# Stop the services
down() {
    print_status "Stopping services..."
    docker-compose down
    print_success "Services stopped!"
}

# Show service status
status() {
    print_status "Service status:"
    docker-compose ps
    echo ""
    print_status "Container logs (last 10 lines):"
    docker-compose logs --tail=10
}

# Show logs
logs() {
    print_status "Showing logs (press Ctrl+C to exit):"
    docker-compose logs -f
}

# Restart services
restart() {
    print_status "Restarting services..."
    docker-compose restart
    print_success "Services restarted!"
}

# Clean up everything
clean() {
    print_warning "This will remove all containers, images, and temporary files."
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Cleaning up..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        sudo rm -rf /tmp/audio_processing
        print_success "Cleanup completed!"
    else
        print_status "Cleanup cancelled."
    fi
}

# Show help
help() {
    echo "Audio Transcription API Management Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build     Build the application containers"
    echo "  up        Start all services"
    echo "  down      Stop all services"
    echo "  status    Show service status and recent logs"
    echo "  logs      Show live logs"
    echo "  restart   Restart all services"
    echo "  clean     Remove all containers and clean up"
    echo "  help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  OLLAMA_URL    - External Ollama URL (default: http://localhost:11434)"
    echo "  OLLAMA_MODEL  - AI model to use (default: llama3)"
    echo ""
    echo "Examples:"
    echo "  ./run.sh build && ./run.sh up"
    echo "  OLLAMA_URL=http://192.168.1.100:11434 ./run.sh up"
    echo "  OLLAMA_MODEL=mistral ./run.sh restart"
}

# Main script logic
case "${1:-help}" in
    build)
        build
        ;;
    up)
        up
        ;;
    down)
        down
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    restart)
        restart
        ;;
    clean)
        clean
        ;;
    help|*)
        help
        ;;
esac 