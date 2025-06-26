#!/usr/bin/env python3
"""
Test script for debugging audio processing pipeline
"""

import logging
import json
import tempfile
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_audio_pipeline():
    """Test the audio processing pipeline with a sample configuration"""
    
    # Load the current steps configuration
    config_file = "settings/steps_config.json"
    
    if not os.path.exists(config_file):
        logger.error(f"Configuration file not found: {config_file}")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        steps_config = config_data.get("steps", {})
        logger.info(f"Loaded configuration with {len(steps_config)} steps")
        
        # Check which audio steps are enabled
        audio_steps = {k: v for k, v in steps_config.items() 
                      if v.get("step_type") == "audio" and v.get("enabled", True)}
        
        logger.info(f"Found {len(audio_steps)} enabled audio steps:")
        for step_id, step_config in audio_steps.items():
            logger.info(f"  - {step_id} (order: {step_config.get('order')}, result_key: {step_config.get('result_key')})")
        
        # Test AudioProcessor initialization
        try:
            from whisper_wrapper import AudioProcessor
            audio_processor = AudioProcessor(steps_config)
            
            processor_info = audio_processor.get_info()
            logger.info(f"AudioProcessor initialized successfully:")
            logger.info(f"  - Backend: {processor_info.get('backend')}")
            logger.info(f"  - Model loaded: {processor_info.get('model_loaded')}")
            logger.info(f"  - Audio steps: {processor_info.get('audio_steps')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AudioProcessor: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return False

def test_whisper_model():
    """Test the Whisper model loading"""
    try:
        from whisper_wrapper import get_whisper_model
        
        whisper_model = get_whisper_model()
        model_info = whisper_model.get_info()
        
        logger.info("Whisper model test:")
        logger.info(f"  - Backend: {model_info.get('backend')}")
        logger.info(f"  - GPU enabled: {model_info.get('gpu_enabled')}")
        logger.info(f"  - Model loaded: {model_info.get('model_loaded')}")
        logger.info(f"  - Diarization available: {model_info.get('diarization_available')}")
        logger.info(f"  - Fast mode: {model_info.get('fast_mode')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to test Whisper model: {e}")
        return False

def test_diarization_engine():
    """Test the diarization engine availability"""
    try:
        from diarization_engine import get_pyannote_engine
        
        engine = get_pyannote_engine()
        if engine:
            logger.info("Diarization engine test:")
            logger.info(f"  - Engine available: True")
            logger.info(f"  - Device: {engine.device}")
            logger.info(f"  - Is loaded: {engine.is_loaded}")
            
            # Try to load models
            try:
                models_loaded = engine.load_models()
                logger.info(f"  - Models loaded: {models_loaded}")
            except Exception as e:
                logger.warning(f"  - Model loading failed: {e}")
        else:
            logger.warning("Diarization engine not available")
        
        return True
        
    except ImportError as e:
        logger.warning(f"Diarization engine import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Diarization engine test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting audio pipeline tests...")
    
    # Test 1: Configuration loading
    logger.info("\n=== Test 1: Configuration Loading ===")
    config_ok = test_audio_pipeline()
    
    # Test 2: Whisper model
    logger.info("\n=== Test 2: Whisper Model ===")
    whisper_ok = test_whisper_model()
    
    # Test 3: Diarization engine
    logger.info("\n=== Test 3: Diarization Engine ===")
    diarization_ok = test_diarization_engine()
    
    # Summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"Configuration: {'✓' if config_ok else '✗'}")
    logger.info(f"Whisper Model: {'✓' if whisper_ok else '✗'}")
    logger.info(f"Diarization: {'✓' if diarization_ok else '✗'}")
    
    if config_ok and whisper_ok:
        logger.info("✅ Basic audio pipeline should work")
    else:
        logger.error("❌ Audio pipeline has issues - check the errors above")
    
    logger.info("Test completed.") 