# 🚀 GPU Deployment Guide

This guide explains how to deploy the Audio Transcription API with GPU acceleration for H100/Kubernetes environments while keeping your MacBook Air development setup intact.

## 📋 Overview

### Current Setup (MacBook Air - UNCHANGED)
- **Docker**: `docker-compose.yml` 
- **Whisper**: CPU-only (`openai-whisper`)
- **Performance**: 2-5x real-time transcription
- **Status**: ✅ **FULLY PRESERVED** - nothing changes!

### GPU Setup (H100/Production - NEW)
- **Docker**: `docker-compose.gpu.yml`
- **Whisper**: GPU-accelerated (`faster-whisper`)
- **Performance**: 50-200x real-time transcription
- **Status**: 🚀 **READY FOR DEPLOYMENT**

## 🛡️ Safety First

**Your current setup is 100% safe!** All GPU optimizations are in separate files:

```
# Your existing files (UNCHANGED)
├── Dockerfile              ← Your current working setup
├── docker-compose.yml      ← Your current working setup  
├── requirements.txt        ← Your current working setup
└── app.py                  ← Your current working setup

# New GPU files (SEPARATE)
├── Dockerfile.gpu          ← GPU-optimized version
├── docker-compose.gpu.yml  ← GPU-optimized version
├── requirements-gpu.txt    ← GPU dependencies
├── config.py               ← Environment detection
├── whisper_wrapper.py      ← Safe GPU/CPU wrapper
└── k8s/                    ← Kubernetes manifests
```

## 🧪 Testing the Setup

### 1. Test Current Setup (Should Still Work)
```bash
# Your existing commands work exactly the same
./run.sh build && ./run.sh up
curl http://localhost:8000/health
```

### 2. Test GPU Preparation
```bash
# Test the new GPU setup without enabling it
python test_gpu_setup.py
```

Expected output:
```
✅ Environment: docker
✅ CPU Count: 8
✅ GPU Available: false
✅ Whisper Backend: openai-whisper
✅ Using your current working CPU setup
```

## 🚀 GPU Deployment Options

### Option 1: Docker Compose (Single Machine)

```bash
# Build GPU-optimized image
docker-compose -f docker-compose.gpu.yml build

# Start with GPU support
docker-compose -f docker-compose.gpu.yml up -d

# Check logs
docker-compose -f docker-compose.gpu.yml logs -f
```

### Option 2: Kubernetes (H100 Cluster)

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment
kubectl get pods -l app=audio-transcription
kubectl logs -l app=audio-transcription
```

## ⚙️ Environment Variables

### Development (MacBook Air)
```bash
# No changes needed - uses defaults
OLLAMA_URL=http://host.docker.internal:11434/api/generate
OLLAMA_MODEL=llama3
```

### Production (H100/GPU)
```bash
# GPU-specific settings
USE_GPU=true
MAX_WORKERS=20
MEMORY_LIMIT_GB=32
OLLAMA_URL=http://ollama-service:11434/api/generate
OLLAMA_MODEL=llama3
```

## 📊 Performance Comparison

| Environment | Transcription Speed | Concurrent Tasks | Memory Usage |
|-------------|-------------------|------------------|--------------|
| **MacBook Air** | 2-5x real-time | 2-4 tasks | 1-2GB |
| **H100/GPU** | 50-200x real-time | 20-50+ tasks | 10-30GB |

## 🔧 Configuration Details

### Automatic Environment Detection

The system automatically detects your environment:

```python
# MacBook Air (CPU mode)
whisper_backend: "openai-whisper"
max_workers: 4
gpu_available: false

# H100 (GPU mode) 
whisper_backend: "faster-whisper"
max_workers: 20
gpu_available: true
```

### Safe Fallbacks

If GPU fails, the system automatically falls back to CPU:

```
GPU mode requested - attempting to load faster-whisper
GPU Whisper failed: CUDA not available - falling back to CPU
✅ CPU Whisper loaded successfully
```

## 🚨 Troubleshooting

### MacBook Air Issues
```bash
# If anything breaks, use your original setup
docker-compose down
docker-compose up -d

# Your original files are unchanged!
```

### GPU Issues
```bash
# Check GPU availability
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi

# Check container logs
docker-compose -f docker-compose.gpu.yml logs audio-altores
```

### Kubernetes Issues
```bash
# Check GPU nodes
kubectl get nodes -l accelerator=nvidia-h100

# Check GPU resources
kubectl describe node <node-name> | grep nvidia.com/gpu

# Check pod status
kubectl describe pod <pod-name>
```

## 🎯 Deployment Checklist

### Pre-Deployment (MacBook Air)
- [ ] Current setup works: `./run.sh up`
- [ ] Test script passes: `python test_gpu_setup.py`
- [ ] Health check works: `curl http://localhost:8000/health`

### GPU Deployment (H100)
- [ ] NVIDIA drivers installed
- [ ] Docker GPU support enabled
- [ ] Kubernetes GPU plugin installed
- [ ] GPU nodes available: `kubectl get nodes -l accelerator=nvidia-h100`

### Post-Deployment
- [ ] GPU containers running: `kubectl get pods`
- [ ] Health check passes: `curl http://<service-ip>:8000/health`
- [ ] Performance monitoring active
- [ ] Rollback plan tested

## 🔄 Rollback Plan

If anything goes wrong in production:

```bash
# Kubernetes rollback
kubectl rollout undo deployment/audio-transcription

# Docker rollback
docker-compose -f docker-compose.gpu.yml down
docker-compose -f docker-compose.yml up -d

# Emergency: Use original setup
git checkout HEAD -- docker-compose.yml Dockerfile requirements.txt
```

## 📈 Monitoring

### Key Metrics to Watch
- **GPU Utilization**: Should be 70-90% during transcription
- **Memory Usage**: Should not exceed 80% of available GPU memory
- **Transcription Speed**: Should be 50-200x real-time
- **Error Rate**: Should be <1%

### Monitoring Commands
```bash
# GPU usage
nvidia-smi -l 1

# Container resources
docker stats

# Kubernetes resources
kubectl top pods
kubectl top nodes
```

## 🎉 Success Indicators

### MacBook Air (Development)
- ✅ `./run.sh up` works as before
- ✅ Transcription completes successfully  
- ✅ No performance regression
- ✅ All existing features work

### H100 (Production)
- ✅ GPU utilization >70% during transcription
- ✅ Transcription speed >50x real-time
- ✅ Multiple concurrent tasks processing
- ✅ No CUDA out-of-memory errors

## 💡 Next Steps

1. **Test locally**: Run `python test_gpu_setup.py`
2. **Verify current setup**: Ensure `./run.sh up` still works
3. **Plan GPU deployment**: Choose Docker Compose or Kubernetes
4. **Monitor performance**: Set up GPU monitoring
5. **Scale gradually**: Start with 1 GPU, then scale up

Remember: **Your current MacBook Air setup remains unchanged and fully functional!** 🎯 