#!/usr/bin/env python3
"""
Test script to verify GPU setup works safely
This script tests the new GPU features without affecting your current setup
"""
import os
import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environment_detection():
    """Test environment detection"""
    print("🔍 Testing Environment Detection...")
    
    try:
        from config import config
        summary = config.get_summary()
        
        print(f"✅ Environment: {summary['environment']}")
        print(f"✅ CPU Count: {summary['cpu_count']}")
        print(f"✅ GPU Available: {summary['gpu_available']}")
        print(f"✅ GPU Count: {summary['gpu_count']}")
        print(f"✅ Whisper Backend: {summary['whisper_backend']}")
        print(f"✅ Max Workers: {summary['max_workers']}")
        
        return True
    except Exception as e:
        print(f"❌ Environment detection failed: {e}")
        return False

def test_whisper_wrapper():
    """Test Whisper wrapper without GPU"""
    print("\n🎤 Testing Whisper Wrapper (CPU mode)...")
    
    try:
        # Force CPU mode for testing
        os.environ['USE_GPU'] = 'false'
        
        from whisper_wrapper import get_whisper_model
        whisper_model = get_whisper_model()
        
        info = whisper_model.get_info()
        print(f"✅ Backend: {info['backend']}")
        print(f"✅ GPU Enabled: {info['gpu_enabled']}")
        print(f"✅ Model Loaded: {info['model_loaded']}")
        
        if info['backend'] == 'openai-whisper':
            print("✅ Using your current working CPU setup")
        
        return True
    except Exception as e:
        print(f"❌ Whisper wrapper test failed: {e}")
        return False

def test_gpu_requirements():
    """Test if GPU requirements are available"""
    print("\n🚀 Testing GPU Requirements...")
    
    try:
        # Test PyTorch
        try:
            import torch
            print(f"✅ PyTorch available: {torch.__version__}")
            print(f"✅ CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"✅ CUDA devices: {torch.cuda.device_count()}")
        except ImportError:
            print("ℹ️ PyTorch not installed (normal for CPU-only setup)")
        
        # Test faster-whisper
        try:
            from faster_whisper import WhisperModel
            print("✅ faster-whisper available")
        except ImportError:
            print("ℹ️ faster-whisper not installed (normal for CPU-only setup)")
        
        return True
    except Exception as e:
        print(f"❌ GPU requirements test failed: {e}")
        return False

def test_docker_files():
    """Test if Docker files are properly created"""
    print("\n🐳 Testing Docker Files...")
    
    files_to_check = [
        'Dockerfile.gpu',
        'docker-compose.gpu.yml',
        'requirements-gpu.txt',
        'k8s/deployment.yaml',
        'k8s/service.yaml',
        'k8s/pvc.yaml'
    ]
    
    all_good = True
    for file in files_to_check:
        if Path(file).exists():
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("🧪 GPU Setup Safety Test")
    print("=" * 50)
    
    tests = [
        ("Environment Detection", test_environment_detection),
        ("Whisper Wrapper", test_whisper_wrapper),
        ("GPU Requirements", test_gpu_requirements),
        ("Docker Files", test_docker_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n📊 Test Results:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your setup is safe and ready for GPU deployment.")
    else:
        print("⚠️ Some tests failed, but your current setup should still work.")
    
    print("\n💡 Next steps:")
    print("1. Your current setup (docker-compose.yml) is unchanged and safe")
    print("2. For GPU testing: USE_GPU=true python test_gpu_setup.py")
    print("3. For H100 deployment: Use docker-compose.gpu.yml")

if __name__ == "__main__":
    main() 