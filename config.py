# config.py
"""
Environment-aware configuration for Audio Transcription API
Safely detects GPU availability and adjusts settings accordingly
"""
import os
import psutil
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class EnvironmentConfig:
    """Safe environment detection with fallbacks"""
    
    def __init__(self):
        # Basic environment detection
        self.is_docker = os.path.exists('/.dockerenv')
        self.is_kubernetes = os.getenv('KUBERNETES_SERVICE_HOST') is not None
        self.cpu_count = psutil.cpu_count()
        
        # GPU detection with safe fallback
        self.gpu_available = False
        self.gpu_count = 0
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
            self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
            if self.gpu_available:
                logger.info(f"GPU detected: {self.gpu_count} device(s)")
        except ImportError:
            logger.info("PyTorch not available - using CPU mode")
        except Exception as e:
            logger.warning(f"GPU detection failed: {e} - falling back to CPU")
    
    @property
    def whisper_backend(self) -> str:
        """Returns the appropriate Whisper backend"""
        if self.gpu_available and os.getenv('USE_GPU', 'false').lower() == 'true':
            return 'faster-whisper'
        return 'openai-whisper'  # Safe default - your current setup
    
    @property
    def whisper_config(self) -> Dict[str, Any]:
        """Returns Whisper configuration based on environment"""
        if self.whisper_backend == 'faster-whisper':
            return {
                'model_size': 'large-v3',
                'device': 'cuda',
                'compute_type': 'float16'
            }
        else:
            # Current working configuration - don't change!
            return {
                'model_size': 'base',
                'device': 'cpu'
            }
    
    @property
    def max_workers(self) -> int:
        """Returns optimal worker count for the environment"""
        # Environment variable override (for testing)
        env_workers = os.getenv('MAX_WORKERS')
        if env_workers:
            return int(env_workers)
        
        # Smart defaults based on environment
        if self.is_kubernetes and self.gpu_available:
            # H100 production environment
            return min(50, self.gpu_count * 10)
        elif self.gpu_available:
            # Single GPU system
            return min(20, self.gpu_count * 8)
        else:
            # CPU-only (your current MacBook Air setup)
            return min(4, self.cpu_count)
    
    @property
    def memory_limit_gb(self) -> int:
        """Returns memory limit based on environment"""
        if self.is_kubernetes:
            # Kubernetes will handle memory limits
            return int(os.getenv('MEMORY_LIMIT_GB', '32'))
        else:
            # Conservative for development
            return 8
    
    def get_summary(self) -> Dict[str, Any]:
        """Returns environment summary for logging"""
        return {
            'environment': 'kubernetes' if self.is_kubernetes else 'docker' if self.is_docker else 'local',
            'cpu_count': self.cpu_count,
            'gpu_available': self.gpu_available,
            'gpu_count': self.gpu_count,
            'whisper_backend': self.whisper_backend,
            'max_workers': self.max_workers,
            'memory_limit_gb': self.memory_limit_gb
        }

# Global config instance
config = EnvironmentConfig()

# Log configuration on import
logger.info(f"Environment configuration: {config.get_summary()}") 