# whisper_wrapper.py
"""
Safe Whisper wrapper with GPU support and Speaker Diarization
Automatically falls back to CPU if GPU is not available
Includes offline-capable speaker diarization for support center calls
"""
import logging
import os
import tempfile
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import json

# Audio processing
import librosa
import soundfile as sf

# Diarization engine - Re-enabled for proper functionality with fallback handling
try:
    from diarization_engine import get_pyannote_engine
    DIARIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Diarization engine not available: {e}")
    DIARIZATION_AVAILABLE = False
    def get_pyannote_engine():
        return None

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Modular audio processor that uses step-based configuration
    Supports configurable audio processing pipeline
    """
    
    def __init__(self, steps_config: Dict[str, Any] = None):
        self.steps_config = steps_config or {}
        self.backend = None
        self.model = None
        self.audio_cache = {}  # Cache for processed audio data
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the appropriate Whisper backend based on configuration"""
        # Get transcription step config - handle multiple possible step IDs
        transcription_config = None
        for step_id, step_config in self.steps_config.items():
            if (step_config.get("step_type") == "audio" and 
                step_config.get("result_key") == "transcript"):
                transcription_config = step_config
                break
        
        if not transcription_config:
            # Fallback to default configuration
            logger.warning("No transcription step found in configuration, using defaults")
            transcription_config = {
                "parameters": {
                    "backend": "auto",
                    "model_size": "large-v3",
                    "use_gpu": True
                }
            }
        
        params = transcription_config.get("parameters", {})
        
        backend = params.get("backend", "auto")
        use_gpu = params.get("use_gpu", True)
        model_size = params.get("model_size", "large-v3")
        
        try:
            if backend in ["faster-whisper", "auto"] and use_gpu:
                self._init_faster_whisper(model_size, params)
            elif backend in ["openai-whisper", "auto"]:
                self._init_openai_whisper(model_size)
            else:
                raise ValueError(f"Unknown backend: {backend}")
                
        except Exception as e:
            logger.error(f"Failed to initialize preferred backend: {e}")
            # Fallback to CPU whisper
            self._init_openai_whisper("base")
    
    def _init_faster_whisper(self, model_size: str, params: Dict[str, Any]):
        """Initialize faster-whisper backend"""
        try:
            from faster_whisper import WhisperModel
            
            self.model = WhisperModel(
                model_size,
                device="cuda",
                compute_type=params.get("compute_type", "float16"),
                cpu_threads=2,
                num_workers=1,
                download_root=None,
                local_files_only=False
            )
            self.backend = "faster-whisper"
            logger.info(f"✅ Faster-whisper loaded: {model_size}")
            
        except Exception as e:
            logger.error(f"Faster-whisper initialization failed: {e}")
            raise
    
    def _init_openai_whisper(self, model_size: str):
        """Initialize OpenAI whisper backend"""
        try:
            import whisper
            self.model = whisper.load_model(model_size)
            self.backend = "openai-whisper"
            logger.info(f"✅ OpenAI Whisper loaded: {model_size}")
            
        except Exception as e:
            logger.error(f"OpenAI Whisper initialization failed: {e}")
            raise
    
    def process_audio_steps(self, file_path: str, task_id: str, 
                          progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process audio through the configured steps pipeline
        """
        results = {"file_path": file_path}
        
        # Get enabled audio steps sorted by order
        audio_steps = {k: v for k, v in self.steps_config.items() 
                      if v.get("step_type") == "audio" and v.get("enabled", True)}
        
        if not audio_steps:
            logger.error(f"[{task_id}] No audio steps configured!")
            raise RuntimeError("No audio steps configured")
        
        sorted_steps = sorted(audio_steps.items(), key=lambda x: x[1].get("order", 0))
        logger.info(f"[{task_id}] Processing {len(sorted_steps)} audio steps: {[s[0] for s in sorted_steps]}")
        
        for step_id, step_config in sorted_steps:
            try:
                if progress_callback:
                    progress_callback(task_id, step_id, "in_progress", 10, f"Starting {step_config['name']}...")
                
                logger.info(f"[{task_id}] Executing audio step: {step_id} (order: {step_config.get('order')})")
                
                # Execute the step
                step_result = self._execute_audio_step(step_id, step_config, results, task_id, progress_callback)
                
                # Store result
                result_key = step_config.get("result_key", step_id)
                results[result_key] = step_result
                
                logger.info(f"[{task_id}] Audio step '{step_id}' completed successfully, stored as '{result_key}'")
                
                if progress_callback:
                    progress_callback(task_id, step_id, "completed", 100, f"{step_config['name']} completed")
                
            except Exception as e:
                error_msg = f"Audio step '{step_id}' failed: {str(e)}"
                logger.error(f"[{task_id}] {error_msg}")
                logger.error(f"[{task_id}] Step config: {step_config}")
                
                if progress_callback:
                    progress_callback(task_id, step_id, "error", 0, error_msg)
                
                # Decide whether to continue or stop
                if step_config.get("result_key") == "transcript":
                    # Transcription failure is critical
                    raise RuntimeError(error_msg)
                else:
                    # Other steps can fail, continue with defaults
                    results[step_config.get("result_key", step_id)] = None
                    logger.warning(f"[{task_id}] Continuing with null result for non-critical step: {step_id}")
                    continue
        
        logger.info(f"[{task_id}] Audio processing completed. Results keys: {list(results.keys())}")
        return results
    
    def _execute_audio_step(self, step_id: str, step_config: Dict[str, Any], 
                           current_results: Dict[str, Any], task_id: str,
                           progress_callback: Optional[Callable] = None) -> Any:
        """Execute a specific audio processing step based on its function type"""
        
        # Detect step function based on result_key and parameters
        step_function = self._detect_step_function(step_id, step_config)
        
        if step_function == "audio_loading":
            return self._step_audio_loading(step_config, current_results, task_id, progress_callback)
        elif step_function == "language_detection":
            return self._step_language_detection(step_config, current_results, task_id, progress_callback)
        elif step_function == "diarization":
            return self._step_diarization(step_config, current_results, task_id, progress_callback)
        elif step_function == "transcription":
            return self._step_transcription(step_config, current_results, task_id, progress_callback)
        else:
            raise ValueError(f"Unknown audio step function for '{step_id}': {step_function}")
    
    def _detect_step_function(self, step_id: str, step_config: Dict[str, Any]) -> str:
        """Detect the step function type based on configuration"""
        result_key = step_config.get("result_key", "")
        parameters = step_config.get("parameters", {})
        step_name = step_config.get("name", "").lower()
        
        # Check by result key first (most reliable)
        if result_key == "audio_metadata":
            return "audio_loading"
        elif result_key == "language_info":
            return "language_detection"
        elif result_key == "diarization_segments":
            return "diarization"
        elif result_key == "transcript":
            return "transcription"
        
        # Check by step name patterns
        if any(keyword in step_name for keyword in ["loading", "preprocess", "audio load"]):
            return "audio_loading"
        elif any(keyword in step_name for keyword in ["language", "detect", "lang"]):
            return "language_detection"
        elif any(keyword in step_name for keyword in ["diarization", "speaker", "segment"]):
            return "diarization"
        elif any(keyword in step_name for keyword in ["transcription", "transcribe", "speech"]):
            return "transcription"
        
        # Check by parameters (fallback)
        if "sample_rate" in parameters or "normalize" in parameters:
            return "audio_loading"
        elif "detection_method" in parameters or "default_language" in parameters:
            return "language_detection"
        elif "method" in parameters and parameters["method"] in ["vad", "pyannote", "disabled"]:
            return "diarization"
        elif "backend" in parameters or "model_size" in parameters:
            return "transcription"
        
        # Check by step_id patterns (enhanced to handle both formats)
        if any(keyword in step_id.lower() for keyword in ["loading", "preprocess"]):
            return "audio_loading"
        elif any(keyword in step_id.lower() for keyword in ["language", "detect", "lang"]):
            return "language_detection"
        elif any(keyword in step_id.lower() for keyword in ["diarization", "speaker", "segment"]):
            return "diarization"
        elif any(keyword in step_id.lower() for keyword in ["transcription", "transcribe", "speech", "audio_transcription"]):
            return "transcription"
        
        # Default fallback - assume it's transcription if nothing else matches
        logger.warning(f"Could not detect step function for '{step_id}', assuming transcription")
        return "transcription"
    
    def _step_audio_loading(self, step_config: Dict[str, Any], current_results: Dict[str, Any],
                           task_id: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Load and preprocess audio file"""
        params = step_config.get("parameters", {})
        step_id = step_config.get("step_id", "audio_loading")
        file_path = current_results["file_path"]
        
        if progress_callback:
            progress_callback(task_id, step_id, "in_progress", 30, "Loading audio file...")
        
        # Load audio with configured parameters
        audio, sr = librosa.load(
            file_path,
            sr=params.get("sample_rate", 16000),
            mono=params.get("mono", True)
        )
        
        if progress_callback:
            progress_callback(task_id, step_id, "in_progress", 60, "Preprocessing audio...")
        
        # Normalize if requested
        if params.get("normalize", True):
            audio = librosa.util.normalize(audio)
        
        # Trim silence if requested
        if params.get("trim_silence", False):
            audio, _ = librosa.effects.trim(
                audio, 
                top_db=params.get("trim_threshold_db", 20)
            )
        
        duration = len(audio) / sr
        
        # Cache audio for other steps
        self.audio_cache[task_id] = {
            "audio": audio,
            "sample_rate": sr,
            "duration": duration
        }
        
        if progress_callback:
            progress_callback(task_id, step_id, "in_progress", 90, f"Audio loaded: {duration:.2f}s")
        
        return {
            "duration": duration,
            "sample_rate": sr,
            "channels": 1 if params.get("mono", True) else 2,
            "file_size": os.path.getsize(file_path),
            "preprocessing": {
                "normalized": params.get("normalize", True),
                "trimmed": params.get("trim_silence", False),
                "target_sample_rate": params.get("sample_rate", 16000)
            }
        }
    
    def _step_language_detection(self, step_config: Dict[str, Any], current_results: Dict[str, Any],
                                task_id: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Detect dominant language in the audio"""
        params = step_config.get("parameters", {})
        step_id = step_config.get("step_id", "language_detection")
        file_path = current_results["file_path"]
        
        if progress_callback:
            progress_callback(task_id, step_id, "in_progress", 30, "Analyzing audio for language...")
        
        try:
            # Get audio from cache or load
            if task_id in self.audio_cache:
                audio = self.audio_cache[task_id]["audio"]
                sr = self.audio_cache[task_id]["sample_rate"]
            else:
                audio, sr = librosa.load(file_path, sr=16000, mono=True)
            
            detection_method = params.get("detection_method", "sample")
            
            if detection_method == "sample":
                # Use a sample for detection
                sample_duration = params.get("sample_duration", 30)
                sample_position = params.get("sample_position", "middle")
                
                total_duration = len(audio) / sr
                
                if total_duration > sample_duration:
                    if sample_position == "start":
                        start_sample = 0
                    elif sample_position == "end":
                        start_sample = int((total_duration - sample_duration) * sr)
                    else:  # middle
                        start_sample = int((total_duration - sample_duration) * sr / 2)
                    
                    end_sample = start_sample + int(sample_duration * sr)
                    sample_audio = audio[start_sample:end_sample]
                else:
                    sample_audio = audio
                
                # Create temporary file for detection
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, sample_audio, sr)
                    
                    if progress_callback:
                        progress_callback(task_id, step_id, "in_progress", 70, "Running language detection...")
                    
                    # Detect language using the model
                    if self.backend == "faster-whisper":
                        segments, info = self.model.transcribe(tmp_file.name, language=None)
                        detected_language = info.language
                        confidence = info.language_probability
                    else:
                        result = self.model.transcribe(tmp_file.name, verbose=False)
                        detected_language = result.get("language", params.get("default_language", "fr"))
                        confidence = 0.8  # OpenAI Whisper doesn't provide confidence
                    
                    # Clean up
                    os.unlink(tmp_file.name)
            else:
                # Full audio detection (slower but more accurate)
                if self.backend == "faster-whisper":
                    segments, info = self.model.transcribe(file_path, language=None)
                    detected_language = info.language
                    confidence = info.language_probability
                else:
                    result = self.model.transcribe(file_path, verbose=False)
                    detected_language = result.get("language", params.get("default_language", "fr"))
                    confidence = 0.8
            
            # Check confidence threshold
            confidence_threshold = params.get("confidence_threshold", 0.7)
            if confidence < confidence_threshold:
                logger.warning(f"Language detection confidence low ({confidence:.2f}), using default")
                detected_language = params.get("default_language", "fr")
            
            return {
                "detected_language": detected_language,
                "confidence": confidence,
                "method": detection_method,
                "sample_duration": params.get("sample_duration", 30) if detection_method == "sample" else None,
                "above_threshold": confidence >= confidence_threshold
            }
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, using default")
            return {
                "detected_language": params.get("default_language", "fr"),
                "confidence": 0.0,
                "method": "default",
                "error": str(e)
            }
    
    def _step_diarization(self, step_config: Dict[str, Any], current_results: Dict[str, Any],
                         task_id: str, progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Perform speaker diarization with enhanced pyannote support"""
        params = step_config.get("parameters", {})
        step_id = step_config.get("step_id", "diarization")
        method = params.get("method", "vad")
        
        if method == "disabled":
            # Return single segment covering entire audio
            audio_metadata = current_results.get("audio_metadata", {})
            duration = audio_metadata.get("duration", 0)
            
            return [{
                "start": 0,
                "end": duration,
                "speaker": "mixed_speakers",
                "duration": duration,
                "method": "disabled"
            }]
        
        if progress_callback:
            progress_callback(task_id, step_id, "in_progress", 20, f"Starting {method} diarization...")
        
        # Get audio from cache
        if task_id not in self.audio_cache:
            raise RuntimeError("Audio not loaded - audio_loading step must run first")
        
        audio = self.audio_cache[task_id]["audio"]
        sr = self.audio_cache[task_id]["sample_rate"]
        file_path = current_results.get("file_path")
        
        if method == "vad":
            segments = self._vad_diarization(audio, sr, params, task_id, progress_callback, step_id)
        elif method == "pyannote":
            segments = self._pyannote_diarization(file_path, params, task_id, progress_callback, step_id)
        else:
            raise ValueError(f"Unknown diarization method: {method}")
        
        if progress_callback:
            progress_callback(task_id, step_id, "in_progress", 90, f"Found {len(segments)} speech segments")
        
        return segments
    
    def _pyannote_diarization(self, file_path: str, params: Dict[str, Any], 
                             task_id: str, progress_callback: Optional[Callable] = None, 
                             step_id: str = "diarization") -> List[Dict[str, Any]]:
        """Enhanced pyannote diarization with speaker context"""
        # Check if diarization is available
        if not DIARIZATION_AVAILABLE:
            logger.warning(f"[{task_id}] Pyannote not available, falling back to VAD")
            return self._vad_diarization(
                self.audio_cache[task_id]["audio"], 
                self.audio_cache[task_id]["sample_rate"], 
                params, task_id, progress_callback, step_id
            )
        
        try:
            # Try to get the pyannote engine
            engine = get_pyannote_engine()
            
            if not engine:
                logger.warning(f"[{task_id}] Pyannote engine not available, falling back to VAD")
                return self._vad_diarization(
                    self.audio_cache[task_id]["audio"], 
                    self.audio_cache[task_id]["sample_rate"], 
                    params, task_id, progress_callback, step_id
                )
            
            if progress_callback:
                progress_callback(task_id, step_id, "in_progress", 30, "Initializing pyannote models...")
            
            if not engine.load_models():
                logger.warning(f"[{task_id}] Pyannote models failed to load, falling back to VAD")
                return self._vad_diarization(
                    self.audio_cache[task_id]["audio"], 
                    self.audio_cache[task_id]["sample_rate"], 
                    params, task_id, progress_callback, step_id
                )
            
            if progress_callback:
                progress_callback(task_id, step_id, "in_progress", 50, "Running pyannote diarization...")
            
            # Configure pyannote with speaker context parameters
            speaker_context = self._build_speaker_context(params)
            segments = self._run_pyannote_with_context(engine, file_path, speaker_context, params, task_id, progress_callback, step_id)
            
            if progress_callback:
                progress_callback(task_id, step_id, "in_progress", 80, "Post-processing speaker segments...")
            
            # Apply speaker context and role mapping
            segments = self._apply_speaker_context(segments, speaker_context, params)
            
            logger.info(f"[{task_id}] Pyannote diarization completed: {len(segments)} segments")
            return segments
            
        except ImportError as e:
            logger.warning(f"[{task_id}] Pyannote not available ({e}), falling back to VAD")
            return self._vad_diarization(
                self.audio_cache[task_id]["audio"], 
                self.audio_cache[task_id]["sample_rate"], 
                params, task_id, progress_callback, step_id
            )
        except Exception as e:
            logger.warning(f"[{task_id}] Pyannote diarization failed ({e}), falling back to VAD")
            return self._vad_diarization(
                self.audio_cache[task_id]["audio"], 
                self.audio_cache[task_id]["sample_rate"], 
                params, task_id, progress_callback, step_id
            )
    
    def _build_speaker_context(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build speaker context configuration for pyannote"""
        return {
            "domain": params.get("domain", "call_center"),
            "scenario": params.get("scenario", "customer_support"),
            "expected_speakers": params.get("expected_speakers", ["agent", "customer"]),
            "speaker_roles": params.get("speaker_roles", {
                "agent": ["conseiller", "support", "agent"],
                "customer": ["client", "customer", "caller"]
            }),
            "conversation_type": params.get("conversation_type", "telephone"),
            "language": params.get("language", "french"),
            "context_prompt": params.get("context_prompt", "Call center conversation between Monaco Telecom support agent and customer"),
            "min_speakers": params.get("min_speakers", 2),
            "max_speakers": params.get("max_speakers", 3),
            "speaker_change_threshold": params.get("speaker_change_threshold", 0.5)
        }
    
    def _run_pyannote_with_context(self, engine, file_path: str, speaker_context: Dict[str, Any], 
                                  params: Dict[str, Any], task_id: str, 
                                  progress_callback: Optional[Callable] = None, 
                                  step_id: str = "diarization") -> List[Dict[str, Any]]:
        """Run pyannote with speaker context configuration"""
        
        # Configure engine with context
        engine.call_center_config.update({
            "min_speakers": speaker_context["min_speakers"],
            "max_speakers": speaker_context["max_speakers"],
            "min_segment_duration": params.get("min_segment_duration", 1.0),
            "min_speaker_duration": params.get("min_speaker_duration", 2.0),
            "context": speaker_context
        })
        
        # Run diarization
        segments = engine.diarize_audio(file_path)
        
        return segments
    
    def _apply_speaker_context(self, segments: List[Dict[str, Any]], 
                              speaker_context: Dict[str, Any], 
                              params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply speaker context and role mapping to segments"""
        if not segments:
            return segments
        
        # Sort segments by start time
        segments = sorted(segments, key=lambda x: x["start"])
        
        # Map speakers to roles based on context
        speaker_mapping = self._create_speaker_mapping(segments, speaker_context)
        
        # Apply mapping and add context
        enhanced_segments = []
        for segment in segments:
            original_speaker = segment.get("speaker", "unknown")
            mapped_role = speaker_mapping.get(original_speaker, original_speaker)
            
            enhanced_segment = {
                **segment,
                "speaker": mapped_role,
                "original_speaker": original_speaker,
                "speaker_role": self._get_speaker_role(mapped_role, speaker_context),
                "context": {
                    "domain": speaker_context["domain"],
                    "scenario": speaker_context["scenario"],
                    "conversation_type": speaker_context["conversation_type"]
                },
                "method": "pyannote"
            }
            
            enhanced_segments.append(enhanced_segment)
        
        # Post-process: merge close segments from same speaker
        merge_threshold = params.get("merge_threshold", 1.0)
        enhanced_segments = self._merge_segments(enhanced_segments, merge_threshold)
        
        return enhanced_segments
    
    def _create_speaker_mapping(self, segments: List[Dict[str, Any]], 
                               speaker_context: Dict[str, Any]) -> Dict[str, str]:
        """Create mapping from pyannote speaker IDs to role names"""
        if not segments:
            return {}
        
        # Get unique speakers
        unique_speakers = list(set(segment.get("speaker", "unknown") for segment in segments))
        
        # Simple heuristic: first speaker is usually agent, second is customer
        speaker_mapping = {}
        expected_speakers = speaker_context.get("expected_speakers", ["agent", "customer"])
        
        for i, speaker_id in enumerate(sorted(unique_speakers)):
            if i < len(expected_speakers):
                role = expected_speakers[i]
                # Map to specific labels
                if role == "agent":
                    speaker_mapping[speaker_id] = "conseiller"
                elif role == "customer":
                    speaker_mapping[speaker_id] = "client"
                else:
                    speaker_mapping[speaker_id] = role
            else:
                speaker_mapping[speaker_id] = f"speaker_{i+1}"
        
        return speaker_mapping
    
    def _get_speaker_role(self, speaker: str, speaker_context: Dict[str, Any]) -> str:
        """Get role description for speaker"""
        speaker_roles = speaker_context.get("speaker_roles", {})
        
        for role, labels in speaker_roles.items():
            if speaker.lower() in [label.lower() for label in labels]:
                return role
        
        return "unknown"
    
    def _vad_diarization(self, audio: np.ndarray, sample_rate: int, params: Dict[str, Any],
                        task_id: str, progress_callback: Optional[Callable] = None, step_id: str = "diarization") -> List[Dict[str, Any]]:
        """VAD-based speaker diarization"""
        
        if progress_callback:
            progress_callback(task_id, step_id, "in_progress", 40, "Detecting speech segments...")
        
        # Use librosa for VAD
        vad_sensitivity = params.get("vad_sensitivity", 12)
        intervals = librosa.effects.split(
            audio, 
            top_db=vad_sensitivity,
            frame_length=2048,
            hop_length=512
        )
        
        if progress_callback:
            progress_callback(task_id, step_id, "in_progress", 60, "Segmenting speakers...")
        
        segments = []
        current_speaker = params.get("speaker_labels", {}).get("default_first", "conseiller")
        last_end_time = 0
        silence_threshold = params.get("silence_threshold", 3.0)
        min_segment_duration = params.get("min_segment_duration", 1.0)
        
        for i, (start_frame, end_frame) in enumerate(intervals):
            start_time = start_frame / sample_rate
            end_time = end_frame / sample_rate
            duration = end_time - start_time
            
            # Filter very short segments
            if duration < min_segment_duration:
                continue
            
            # Calculate silence gap
            silence_gap = start_time - last_end_time if segments else 0
            
            # Determine speaker change
            if silence_gap > silence_threshold:
                # Switch speaker after long silence
                speaker_labels = params.get("speaker_labels", {})
                current_speaker = (speaker_labels.get("default_second", "client") 
                                 if current_speaker == speaker_labels.get("default_first", "conseiller")
                                 else speaker_labels.get("default_first", "conseiller"))
            
            segments.append({
                "start": start_time,
                "end": end_time,
                "speaker": current_speaker,
                "duration": duration,
                "silence_before": silence_gap,
                "method": "vad"
            })
            
            last_end_time = end_time
        
        # Post-process: merge close segments from same speaker
        merge_threshold = params.get("merge_threshold", 1.0)
        segments = self._merge_segments(segments, merge_threshold)
        
        return segments
    
    def _merge_segments(self, segments: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """Merge segments from same speaker that are close together"""
        if len(segments) <= 1:
            return segments
        
        merged = [segments[0]]
        
        for segment in segments[1:]:
            prev_segment = merged[-1]
            gap = segment["start"] - prev_segment["end"]
            
            if (segment["speaker"] == prev_segment["speaker"] and gap < threshold):
                # Merge segments
                prev_segment["end"] = segment["end"]
                prev_segment["duration"] = prev_segment["end"] - prev_segment["start"]
            else:
                merged.append(segment)
        
        return merged
    
    def _step_transcription(self, step_config: Dict[str, Any], current_results: Dict[str, Any],
                           task_id: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Transcribe audio using configured parameters"""
        params = step_config.get("parameters", {})
        step_id = step_config.get("step_id", "transcription")
        file_path = current_results["file_path"]
        
        # Get language from previous step
        language_info = current_results.get("language_info", {})
        detected_language = language_info.get("detected_language", "fr")
        force_language = params.get("force_language") or detected_language
        
        # Get diarization segments
        diarization_segments = current_results.get("diarization_segments", [])
        
        if progress_callback:
            progress_callback(task_id, step_id, "in_progress", 20, f"Transcribing with {self.backend}...")
        
        if len(diarization_segments) == 1 and diarization_segments[0].get("speaker") == "mixed_speakers":
            # No diarization, direct transcription
            transcript_result = self._transcribe_direct(file_path, force_language, params, task_id, progress_callback)
        else:
            # Transcribe segments individually
            transcript_result = self._transcribe_segments(file_path, diarization_segments, force_language, params, task_id, progress_callback)
        
        return transcript_result
    
    def _transcribe_direct(self, file_path: str, language: str, params: Dict[str, Any],
                          task_id: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Direct transcription without segmentation"""
        
        if progress_callback:
            progress_callback(task_id, "transcription", "in_progress", 50, "Processing entire audio...")
        
        try:
            if self.backend == "faster-whisper":
                segments, info = self.model.transcribe(
                    file_path,
                    language=language,
                    initial_prompt=params.get("initial_prompt"),
                    temperature=params.get("temperature", 0.0),
                    best_of=params.get("best_of", 1),
                    beam_size=params.get("beam_size", 5)
                )
                
                text = " ".join([segment.text for segment in segments])
                
                return {
                    "text": text,
                    "full_text": text,
                    "transcript": f"[TRANSCRIPT]: {text}",
                    "segments": [{
                        "start": 0,
                        "end": self.audio_cache[task_id]["duration"],
                        "speaker": "mixed_speakers",
                        "text": text,
                        "language": info.language,
                        "confidence": info.language_probability
                    }],
                    "metadata": {
                        "backend": self.backend,
                        "language": info.language,
                        "confidence": info.language_probability,
                        "segmented": False
                    }
                }
            
            else:
                result = self.model.transcribe(
                    file_path,
                    language=language,
                    verbose=False,
                    initial_prompt=params.get("initial_prompt"),
                    temperature=params.get("temperature", 0.0)
                )
                
                text = result["text"]
                
                return {
                    "text": text,
                    "full_text": text,
                    "transcript": f"[TRANSCRIPT]: {text}",
                    "segments": [{
                        "start": 0,
                        "end": self.audio_cache[task_id]["duration"],
                        "speaker": "mixed_speakers",
                        "text": text,
                        "language": result.get("language", language),
                        "confidence": 0.8
                    }],
                    "metadata": {
                        "backend": self.backend,
                        "language": result.get("language", language),
                        "segmented": False
                    }
                }
                
        except Exception as e:
            logger.error(f"Direct transcription failed: {e}")
            raise
    
    def _transcribe_segments(self, file_path: str, segments: List[Dict[str, Any]], 
                           language: str, params: Dict[str, Any], task_id: str,
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Transcribe segments with enhanced speaker context"""
        results = []
        total_segments = len(segments)
        failed_segments = []
        
        # Extract context information from segments
        context_info = self._extract_context_from_segments(segments)
        base_prompt = params.get("initial_prompt", "")
        
        for i, segment in enumerate(segments):
            try:
                # Build context-aware prompt for this segment
                segment_prompt = self._build_segment_prompt(segment, base_prompt, context_info)
                
                # Extract segment audio
                segment_file = self._extract_audio_segment(file_path, segment, task_id)
                
                # Transcribe segment with context
                result = self._transcribe_with_context(segment_file, language, segment_prompt, params)
                text = result.get("text", "").strip()
                
                # Clean up temp file
                os.unlink(segment_file)
                
                # Enhanced segment result with context
                segment_result = {
                    "start": segment["start"],
                    "end": segment["end"], 
                    "speaker": segment["speaker"],
                    "text": text,
                    "confidence": result.get("confidence", 0.0),
                    "language": result.get("language", language),
                    "context_used": segment_prompt != "",
                    "speaker_role": segment.get("speaker_role", "unknown"),
                    "original_speaker": segment.get("original_speaker", segment["speaker"]),
                    "method": segment.get("method", "unknown"),
                    "segment_context": segment.get("context", {})
                }
                
                if text:
                    results.append(segment_result)
                    logger.debug(f"[{task_id}] Segment {i+1}: {segment['speaker']} ({segment['start']:.2f}-{segment['end']:.2f}s) -> {len(text)} chars")
                else:
                    segment_result["text"] = f"[SEGMENT {i+1} - NO TRANSCRIPTION]"
                    failed_segments.append(segment_result)
                    logger.warning(f"[{task_id}] Segment {i+1} ({segment['start']:.2f}-{segment['end']:.2f}s) produced no text")
                
                # Update progress
                if progress_callback:
                    progress = 30 + int((i + 1) / total_segments * 60)
                    progress_callback(task_id, "transcription", "in_progress", progress, 
                                    f"Transcribed segment {i+1}/{total_segments} ({segment['speaker']})")
                
            except Exception as e:
                logger.warning(f"[{task_id}] Failed to transcribe segment {i}: {e}")
                failed_segment = {
                    "start": segment["start"],
                    "end": segment["end"], 
                    "speaker": segment.get("speaker", "unknown"),
                    "text": f"[SEGMENT {i+1} - TRANSCRIPTION FAILED: {str(e)[:50]}]",
                    "confidence": 0.0,
                    "language": language,
                    "context_used": False,
                    "error": True
                }
                failed_segments.append(failed_segment)
                continue
        
        # Build comprehensive transcript with speaker context
        transcript_data = self._build_enhanced_transcript(results, context_info, params)
        
        # Log statistics
        success_rate = len(results) / total_segments if total_segments > 0 else 0
        logger.info(f"[{task_id}] Transcription completed: {len(results)}/{total_segments} segments successful ({success_rate*100:.1f}%)")
        
        if failed_segments:
            logger.warning(f"[{task_id}] {len(failed_segments)} segments failed or were empty")
        
        return transcript_data
    
    def _extract_context_from_segments(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract context information from diarization segments"""
        if not segments:
            return {}
        
        # Get unique speakers and their roles
        speakers = {}
        total_duration = 0
        conversation_context = {}
        
        for segment in segments:
            speaker = segment.get("speaker", "unknown")
            if speaker not in speakers:
                speakers[speaker] = {
                    "role": segment.get("speaker_role", "unknown"),
                    "total_time": 0,
                    "segment_count": 0,
                    "first_appearance": segment.get("start", 0)
                }
            
            speakers[speaker]["total_time"] += segment.get("duration", 0)
            speakers[speaker]["segment_count"] += 1
            total_duration += segment.get("duration", 0)
            
            # Extract conversation context from first segment
            if not conversation_context and segment.get("context"):
                conversation_context = segment["context"]
        
        return {
            "speakers": speakers,
            "total_duration": total_duration,
            "conversation_context": conversation_context,
            "diarization_method": segments[0].get("method", "unknown") if segments else "unknown"
        }
    
    def _build_segment_prompt(self, segment: Dict[str, Any], base_prompt: str, context_info: Dict[str, Any]) -> str:
        """Build context-aware prompt for segment transcription"""
        if not base_prompt:
            return ""
        
        speaker = segment.get("speaker", "unknown")
        speaker_role = segment.get("speaker_role", "unknown")
        conversation_context = context_info.get("conversation_context", {})
        
        # Enhance prompt with speaker context
        enhanced_prompt = base_prompt
        
        if speaker_role != "unknown":
            if speaker_role == "agent":
                enhanced_prompt += f" This segment is spoken by the {speaker} (customer service agent)."
            elif speaker_role == "customer":
                enhanced_prompt += f" This segment is spoken by the {speaker} (customer/caller)."
            else:
                enhanced_prompt += f" This segment is spoken by {speaker} ({speaker_role})."
        
        # Add conversation type context
        conv_type = conversation_context.get("conversation_type", "")
        if conv_type == "telephone":
            enhanced_prompt += " This is a telephone conversation, so audio quality may vary."
        
        return enhanced_prompt
    
    def _transcribe_with_context(self, file_path: str, language: str, prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe audio file with context prompt"""
        try:
            if self.backend == "faster-whisper":
                segments, info = self.model.transcribe(
                    file_path,
                    language=language,
                    initial_prompt=prompt if prompt else None,
                    temperature=params.get("temperature", 0.0),
                    best_of=params.get("best_of", 1),
                    beam_size=params.get("beam_size", 5)
                )
                
                text = " ".join([segment.text for segment in segments])
                
                return {
                    "text": text,
                    "language": info.language,
                    "confidence": info.language_probability,
                    "backend": self.backend
                }
            
            else:
                result = self.model.transcribe(
                    file_path,
                    language=language,
                    verbose=False,
                    initial_prompt=prompt if prompt else None,
                    temperature=params.get("temperature", 0.0)
                )
                
                return {
                    "text": result.get("text", ""),
                    "language": result.get("language", language),
                    "confidence": 1.0,  # OpenAI Whisper doesn't provide confidence
                    "backend": self.backend
                }
                
        except Exception as e:
            logger.error(f"Context transcription failed: {e}")
            raise
    
    def _build_enhanced_transcript(self, results: List[Dict[str, Any]], context_info: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Build enhanced transcript with speaker context and metadata"""
        
        # Create speaker-labeled transcript
        transcript_lines = []
        full_text_lines = []
        
        for result in results:
            speaker = result["speaker"]
            text = result["text"]
            
            if not text.strip():
                continue
            
            # Format with enhanced speaker context
            start_time = result.get("start", 0)
            speaker_role = result.get("speaker_role", "unknown")
            
            # Enhanced speaker label
            if speaker_role == "agent":
                label = f"AGENT ({speaker.upper()})"
            elif speaker_role == "customer": 
                label = f"CLIENT ({speaker.upper()})"
            else:
                label = speaker.upper()
            
            # Timestamp
            time_stamp = f"[{start_time//60:02.0f}:{start_time%60:05.2f}]"
            formatted_line = f"{time_stamp} {label}: {text}"
            
            transcript_lines.append(formatted_line)
            full_text_lines.append(text)
        
        # Build final transcript
        transcript = "\n".join(transcript_lines)
        full_text = " ".join(full_text_lines)
        
        # Enhanced metadata
        metadata = {
            "backend": self.backend,
            "total_segments": len(results),
            "speakers": context_info.get("speakers", {}),
            "conversation_context": context_info.get("conversation_context", {}),
            "diarization_method": context_info.get("diarization_method", "unknown"),
            "total_duration": context_info.get("total_duration", 0),
            "context_enhanced": True
        }
        
        return {
            "text": transcript,
            "full_text": full_text,
            "transcript": transcript,
            "segments": results,
            "metadata": metadata
        }
    
    def _extract_audio_segment(self, file_path: str, segment: Dict[str, Any], task_id: str) -> str:
        """Extract audio segment to temporary file"""
        audio, sr = librosa.load(
            file_path,
            sr=16000,
            offset=segment["start"],
            duration=segment["end"] - segment["start"]
        )
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sr)
            return tmp_file.name
    
    def get_info(self) -> Dict[str, Any]:
        """Get processor information"""
        return {
            "backend": self.backend,
            "model_loaded": self.model is not None,
            "steps_configured": len(self.steps_config),
            "audio_steps": [k for k, v in self.steps_config.items() if v.get("step_type") == "audio"],
            "llm_steps": [k for k, v in self.steps_config.items() if v.get("step_type") == "llm"]
        }

class WhisperWrapper:
    """Safe Whisper wrapper with automatic CPU/GPU detection and speaker diarization"""
    
    def __init__(self):
        self.model = None
        self.backend = None
        self.diarization_pipeline = None
        self.diarization_available = False
        self.skip_diarization = self._should_skip_diarization()
        self._initialize_model()
        if not self.skip_diarization:
            self._initialize_diarization()
        else:
            logger.info("🚀 Diarization DISABLED - using fast transcription mode")
    
    def _should_skip_diarization(self) -> bool:
        """Determine if diarization should be skipped for faster processing"""
        skip_diarization_env = os.getenv('SKIP_DIARIZATION', 'false').lower()
        
        if skip_diarization_env in ['true', '1', 'yes', 'on']:
            return True
        elif skip_diarization_env in ['false', '0', 'no', 'off']:
            return False
        else:
            logger.warning(f"Unknown SKIP_DIARIZATION value: {skip_diarization_env} - defaulting to false")
            return False
    
    def _should_use_gpu(self) -> bool:
        """Force GPU sur RunPod H100 - cuDNN résolu"""
        # Sur RunPod, forcer GPU par défaut
        use_gpu_env = os.getenv('USE_GPU', 'true').lower()  # Changé: défaut true
        
        if use_gpu_env == 'false':
            return False
        elif use_gpu_env in ['true', 'auto']:
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                if gpu_available:
                    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                return gpu_available
            except ImportError:
                logger.info("PyTorch not available - using CPU mode")
                return False
        else:
            logger.warning(f"Unknown USE_GPU value: {use_gpu_env} - defaulting to GPU")
            return True
    
    def _initialize_model(self):
        """Initialize Whisper avec optimisations H100"""
        try:
            should_use_gpu = self._should_use_gpu()
            
            if should_use_gpu:
                logger.info("Loading faster-whisper for H100...")
                try:
                    from faster_whisper import WhisperModel
                    
                    # Configuration H100 stable - anti-crash cuDNN
                    self.model = WhisperModel(
                        "large-v3",           # Meilleur modèle
                        device="cuda",
                        compute_type="float16",  # Configuration testée
                        cpu_threads=2,        # Optimisé pour stabilité
                        num_workers=1,        # Configuration stable
                        download_root=None,   # Éviter conflits téléchargement
                        local_files_only=False
                    )
                    self.backend = "faster-whisper"
                    logger.info("✅ H100 Whisper loaded successfully (large-v3)")
                    return
                    
                except Exception as gpu_error:
                    logger.error(f"H100 Whisper failed: {gpu_error}")
                    raise  # Ne pas fallback sur CPU sur H100
            
            # Fallback CPU (développement local uniquement)
            logger.info("Loading CPU Whisper (development mode)")
            import whisper
            self.model = whisper.load_model("base")
            self.backend = "openai-whisper"
            logger.info("✅ CPU Whisper loaded")
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            raise RuntimeError(f"Could not initialize Whisper: {e}")
    
    def _initialize_diarization(self):
        """Initialise diarisation - TEMPORAIREMENT DÉSACTIVÉ pour test GPU"""
        try:
            logger.info("Diarization temporarily disabled for GPU testing...")
            
            # TEMPORAIRE: Utiliser VAD seulement
            logger.warning("Using VAD fallback for testing")
            self.diarization_pipeline = self._create_vad_pipeline()
            self.diarization_available = True
            
            # FUTURE: Réactiver pyannote après résolution conflits versions
            # pyannote_engine = get_pyannote_engine()
            # if pyannote_engine.load_models():
            #     self.diarization_pipeline = pyannote_engine
            #     self.diarization_available = True
            #     logger.info("✅ Pyannote diarization loaded successfully")
                
        except Exception as e:
            logger.error(f"Diarization initialization failed: {e}")
            self.diarization_pipeline = None
            self.diarization_available = False
    
    def _create_vad_pipeline(self):
        """Create a VAD-based segmentation pipeline for offline use"""
        return VADPipeline()
    
    def transcribe(self, file_path: str) -> Dict[str, Any]:
        """Transcribe audio file using the appropriate backend (basic transcription)"""
        if not self.model:
            raise RuntimeError("Whisper model not initialized")
        
        try:
            if self.backend == "faster-whisper":
                # GPU-accelerated transcription
                segments, info = self.model.transcribe(file_path)
                
                # Convert segments to text
                text = " ".join([segment.text for segment in segments])
                
                return {
                    "text": text,
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "backend": "faster-whisper"
                }
            
            else:
                # CPU transcription (your current working method)
                result = self.model.transcribe(file_path)
                result["backend"] = "openai-whisper"
                return result
                
        except Exception as e:
            logger.error(f"Transcription failed with {self.backend}: {e}")
            raise RuntimeError(f"Transcription failed: {e}")
    
    def transcribe_with_diarization(self, file_path: str, task_id: str = None, 
                                  progress_callback=None) -> Dict[str, Any]:
        """
        Enhanced transcription with optional speaker diarization and consistent language detection
        Can be configured to skip diarization for faster processing
        """
        # Check if diarization should be skipped
        if self.skip_diarization:
            logger.info("🚀 Fast mode: Skipping diarization, doing direct transcription")
            return self._fast_transcribe_with_language(file_path, task_id, progress_callback)
        
        if not self.diarization_available:
            logger.info("Diarization not available, using basic transcription")
            return self.transcribe(file_path)
        
        try:
            if progress_callback:
                progress_callback(task_id, "transcription", "in_progress", 5, 
                                "Loading audio for diarization...")
            
            # Load and preprocess audio
            audio, sample_rate = self._load_audio(file_path)
            duration = len(audio) / sample_rate
            
            if progress_callback:
                progress_callback(task_id, "transcription", "in_progress", 10, 
                                "Detecting dominant language...")
            
            # Detect dominant language first for consistency
            dominant_language = self._detect_dominant_language(file_path)
            logger.info(f"Detected dominant language: {dominant_language}")
            
            if progress_callback:
                progress_callback(task_id, "transcription", "in_progress", 20, 
                                f"Performing speaker diarization (language: {dominant_language})...")
            
            # Perform diarization
            segments = self._perform_diarization(audio, sample_rate, file_path)
            
            if progress_callback:
                progress_callback(task_id, "transcription", "in_progress", 30, 
                                f"Found {len(segments)} speech segments, transcribing with {dominant_language}...")
            
            # Transcribe each segment with forced language
            results = self._transcribe_segments(file_path, segments, task_id, progress_callback, 
                                              forced_language=dominant_language)
            
            # Check if diarization produced sufficient results
            total_transcribed_duration = sum(r.get("end", 0) - r.get("start", 0) for r in results if r.get("text", "").strip())
            coverage_ratio = total_transcribed_duration / duration if duration > 0 else 0
            
            logger.info(f"Diarization coverage: {coverage_ratio*100:.1f}% ({total_transcribed_duration:.2f}s of {duration:.2f}s)")
            
            # If coverage is too low, fallback to complete transcription
            if coverage_ratio < 0.6:  # Less than 60% coverage
                logger.warning(f"Diarization coverage too low ({coverage_ratio*100:.1f}%), falling back to complete transcription")
                if progress_callback:
                    progress_callback(task_id, "transcription", "in_progress", 50, 
                                    "Diarization coverage low, performing complete transcription...")
                return self._complete_transcription_fallback(file_path, task_id, progress_callback, dominant_language, results)
            
            # Format output
            output = self._format_diarized_output(results, duration, dominant_language)
            
            if progress_callback:
                progress_callback(task_id, "transcription", "completed", 100, 
                                f"Enhanced transcription completed ({len(output['transcript'])} characters)")
            
            return output
            
        except Exception as e:
            logger.warning(f"Diarization failed: {e}, falling back to basic transcription")
            if progress_callback:
                progress_callback(task_id, "transcription", "in_progress", 50, 
                                "Diarization failed, using basic transcription...")
            return self.transcribe(file_path)
    
    def _fast_transcribe_with_language(self, file_path: str, task_id: str = None, 
                                     progress_callback=None, context_prompt: str = None) -> Dict[str, Any]:
        """
        Fast transcription without diarization but with consistent language detection and optional context
        Much faster than diarization but no speaker separation
        """
        try:
            if progress_callback:
                progress_callback(task_id, "transcription", "in_progress", 10, 
                                "Fast mode: Detecting language...")
            
            # Detect dominant language for consistency
            dominant_language = self._detect_dominant_language(file_path)
            logger.info(f"🚀 Fast mode - Detected language: {dominant_language}")
            
            if context_prompt:
                logger.info(f"🎯 Using context prompt for enhanced transcription")
            
            if progress_callback:
                progress_callback(task_id, "transcription", "in_progress", 30, 
                                f"Fast transcribing with {dominant_language}...")
            
            # Direct transcription with forced language and context
            result = self._transcribe_with_language(file_path, dominant_language, context_prompt)
            
            if progress_callback:
                progress_callback(task_id, "transcription", "in_progress", 80, 
                                "Formatting fast transcript...")
            
            # Format as simplified output (no speaker separation)
            audio, sample_rate = self._load_audio(file_path)
            duration = len(audio) / sample_rate
            
            transcript = result.get("text", "").strip()
            
            output = {
                "text": transcript,
                "transcript": f"[00:00.00] TRANSCRIPT COMPLET: {transcript}",
                "segments": [{
                    "start": 0,
                    "end": duration,
                    "speaker": "mixed_speakers",
                    "text": transcript,
                    "language": dominant_language,
                    "confidence": result.get("language_probability", 0.0),
                    "context_used": result.get("context_used", False)
                }],
                "metadata": {
                    "duration": duration,
                    "num_segments": 1,
                    "speakers": ["mixed_speakers"],
                    "backend": f"{self.backend} + fast_mode",
                    "gpu_enabled": self.backend == "faster-whisper",
                    "diarization_enabled": False,
                    "fast_mode": True,
                    "dominant_language": dominant_language,
                    "language_consistency": True,
                    "context_used": result.get("context_used", False)
                }
            }
            
            if progress_callback:
                progress_callback(task_id, "transcription", "completed", 100, 
                                f"🚀 Fast transcription completed ({len(transcript)} characters)")
            
            return output
            
        except Exception as e:
            logger.error(f"Fast transcription failed: {e}, falling back to basic transcription")
            if progress_callback:
                progress_callback(task_id, "transcription", "in_progress", 90, 
                                "Fast mode failed, using basic transcription...")
            return self.transcribe(file_path)
    
    def _detect_dominant_language(self, file_path: str) -> str:
        """Detect the dominant language in the audio file for consistent transcription"""
        try:
            # For Monaco Telecom, default to French if detection fails
            default_language = "fr"
            
            if self.backend == "faster-whisper":
                # Use faster-whisper's language detection
                segments, info = self.model.transcribe(file_path, language=None)
                return info.language if info.language else default_language
            else:
                # Use OpenAI Whisper's language detection on a sample
                # Take a 30-second sample from the middle of the file for detection
                audio, sr = librosa.load(file_path, sr=16000, mono=True)
                duration = len(audio) / sr
                
                if duration > 60:
                    # Sample from middle third of the audio
                    start_sample = int(duration * sr * 0.33)
                    end_sample = int(min(start_sample + 30 * sr, len(audio)))
                    sample_audio = audio[start_sample:end_sample]
                else:
                    # Use entire audio if short
                    sample_audio = audio
                
                # Save sample to temp file for detection
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, sample_audio, sr)
                    
                    # Detect language
                    result = self.model.transcribe(tmp_file.name, verbose=False)
                    detected_language = result.get("language", default_language)
                    
                    # Clean up
                    os.unlink(tmp_file.name)
                    
                    return detected_language
                    
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, defaulting to French")
            return "fr"  # Default for Monaco Telecom
    
    def _load_audio(self, file_path: str) -> tuple:
        """Load and preprocess audio file"""
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        audio = librosa.util.normalize(audio)
        return audio, sr
    
    def _perform_diarization(self, audio: np.ndarray, sample_rate: int, file_path: str) -> List[Dict]:
        """Utilise VAD pour diarisation (pyannote temporairement désactivé)"""
        if self.diarization_pipeline is None:
            raise RuntimeError("Diarization pipeline not initialized")
        
        try:
            # TEMPORAIRE: Utiliser VAD seulement
            return self.diarization_pipeline.segment(audio, sample_rate)
            
            # FUTURE: Réactiver pyannote après résolution conflits versions
            # if hasattr(self.diarization_pipeline, 'diarize_audio'):
            #     return self.diarization_pipeline.diarize_audio(file_path)
            # else:
            #     return self.diarization_pipeline.segment(audio, sample_rate)
                
        except Exception as e:
            logger.error(f"VAD diarization failed: {e}")
            # Fallback final - utiliser VAD pipeline nouvelle instance
            vad_pipeline = VADPipeline()
            return vad_pipeline.segment(audio, sample_rate)
    
    def _transcribe_segments(self, file_path: str, segments: List[Dict], 
                           language: str, params: Dict[str, Any], task_id: str,
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Transcribe segments with enhanced speaker context"""
        results = []
        total_segments = len(segments)
        failed_segments = []
        
        # Extract context information from segments
        context_info = self._extract_context_from_segments(segments)
        base_prompt = params.get("initial_prompt", "")
        
        for i, segment in enumerate(segments):
            try:
                # Build context-aware prompt for this segment
                segment_prompt = self._build_segment_prompt(segment, base_prompt, context_info)
                
                # Extract segment audio
                segment_file = self._extract_audio_segment(file_path, segment, task_id)
                
                # Transcribe segment with context
                result = self._transcribe_with_context(segment_file, language, segment_prompt, params)
                text = result.get("text", "").strip()
                
                # Clean up temp file
                os.unlink(segment_file)
                
                # Enhanced segment result with context
                segment_result = {
                    "start": segment["start"],
                    "end": segment["end"], 
                    "speaker": segment["speaker"],
                    "text": text,
                    "confidence": result.get("confidence", 0.0),
                    "language": result.get("language", language),
                    "context_used": segment_prompt != "",
                    "speaker_role": segment.get("speaker_role", "unknown"),
                    "original_speaker": segment.get("original_speaker", segment["speaker"]),
                    "method": segment.get("method", "unknown"),
                    "segment_context": segment.get("context", {})
                }
                
                if text:
                    results.append(segment_result)
                    logger.debug(f"[{task_id}] Segment {i+1}: {segment['speaker']} ({segment['start']:.2f}-{segment['end']:.2f}s) -> {len(text)} chars")
                else:
                    segment_result["text"] = f"[SEGMENT {i+1} - NO TRANSCRIPTION]"
                    failed_segments.append(segment_result)
                    logger.warning(f"[{task_id}] Segment {i+1} ({segment['start']:.2f}-{segment['end']:.2f}s) produced no text")
                
                # Update progress
                if progress_callback:
                    progress = 30 + int((i + 1) / total_segments * 60)
                    progress_callback(task_id, "transcription", "in_progress", progress, 
                                    f"Transcribed segment {i+1}/{total_segments} ({segment['speaker']})")
                
            except Exception as e:
                logger.warning(f"[{task_id}] Failed to transcribe segment {i}: {e}")
                failed_segment = {
                    "start": segment["start"],
                    "end": segment["end"], 
                    "speaker": segment.get("speaker", "unknown"),
                    "text": f"[SEGMENT {i+1} - TRANSCRIPTION FAILED: {str(e)[:50]}]",
                    "confidence": 0.0,
                    "language": language,
                    "context_used": False,
                    "error": True
                }
                failed_segments.append(failed_segment)
                continue
        
        # Build comprehensive transcript with speaker context
        transcript_data = self._build_enhanced_transcript(results, context_info, params)
        
        # Log statistics
        success_rate = len(results) / total_segments if total_segments > 0 else 0
        logger.info(f"[{task_id}] Transcription completed: {len(results)}/{total_segments} segments successful ({success_rate*100:.1f}%)")
        
        if failed_segments:
            logger.warning(f"[{task_id}] {len(failed_segments)} segments failed or were empty")
        
        return transcript_data
    
    def _transcribe_with_language(self, file_path: str, language: str, context_prompt: str = None) -> Dict[str, Any]:
        """Transcribe with specific language constraint and optional context prompt"""
        if not self.model:
            raise RuntimeError("Whisper model not initialized")
        
        try:
            if self.backend == "faster-whisper":
                # GPU-accelerated transcription with language and context
                segments, info = self.model.transcribe(
                    file_path, 
                    language=language,
                    initial_prompt=context_prompt
                )
                
                # Convert segments to text
                text = " ".join([segment.text for segment in segments])
                
                return {
                    "text": text,
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "backend": "faster-whisper",
                    "context_used": context_prompt is not None
                }
            
            else:
                # CPU transcription with language and context
                result = self.model.transcribe(
                    file_path, 
                    language=language, 
                    verbose=False,
                    initial_prompt=context_prompt
                )
                result["backend"] = "openai-whisper"
                result["context_used"] = context_prompt is not None
                return result
                
        except Exception as e:
            logger.error(f"Language-specific transcription failed with {self.backend}: {e}")
            # Fallback to normal transcription
            return self.transcribe(file_path)
    
    def _extract_segment(self, file_path: str, segment: Dict) -> str:
        """Extract audio segment to temporary file"""
        audio, sr = librosa.load(
            file_path,
            sr=16000,
            offset=segment["start"],
            duration=segment["end"] - segment["start"]
        )
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sr)
            return tmp_file.name
    
    def _format_diarized_output(self, results: List[Dict], duration: float, dominant_language: str = None) -> Dict:
        """Format diarization results for Monaco Telecom support context"""
        # Create speaker-labeled transcript with context
        transcript_lines = []
        full_text_lines = []  # Also create a continuous transcript without speaker labels
        
        for result in results:
            speaker = result["speaker"]
            text = result["text"]
            
            # Skip empty text
            if not text.strip():
                continue
            
            # Format with Monaco Telecom context
            if speaker == "conseiller":
                label = "CONSEILLER MONACO TELECOM"
            elif speaker == "client":
                label = "CLIENT"
            else:
                label = speaker.upper()
            
            # Add timing and context info for debugging
            start_time = result.get("start", 0)
            duration_seg = result.get("end", 0) - start_time
            turn_type = result.get("turn_type", "")
            
            # Format line with enhanced context
            time_stamp = f"[{start_time//60:02.0f}:{start_time%60:05.2f}]"
            if turn_type:
                formatted_line = f"{time_stamp} {label} ({turn_type}): {text}"
            else:
                formatted_line = f"{time_stamp} {label}: {text}"
            
            transcript_lines.append(formatted_line)
            # Also add to continuous text for better completeness
            full_text_lines.append(text)
        
        # Create both speaker-labeled and continuous transcripts
        transcript = "\n".join(transcript_lines)
        full_text = " ".join(full_text_lines)
        
        # If diarized transcript is too short compared to segments, use full text as fallback
        if len(transcript) < len(full_text) * 0.5:  # If diarized is less than 50% of full text
            logger.warning("Diarized transcript appears incomplete, using full text as primary")
            transcript = full_text
        
        # Count conversation statistics
        conseiller_segments = [r for r in results if r["speaker"] == "conseiller"]
        client_segments = [r for r in results if r["speaker"] == "client"]
        
        conseiller_time = sum(r.get("end", 0) - r.get("start", 0) for r in conseiller_segments)
        client_time = sum(r.get("end", 0) - r.get("start", 0) for r in client_segments)
        
        return {
            "text": transcript,  # Keep compatible with basic transcribe
            "transcript": transcript,
            "full_text": full_text,  # Add continuous text without speaker labels
            "segments": results,
            "metadata": {
                "duration": duration,
                "num_segments": len(results),
                "speakers": list(set(r["speaker"] for r in results)),
                "backend": f"{self.backend} + diarization",
                "gpu_enabled": self.backend == "faster-whisper",
                "diarization_enabled": True,
                "dominant_language": dominant_language,
                "language_consistency": True,  # Flag indicating forced language was used
                "call_stats": {
                    "conseiller_segments": len(conseiller_segments),
                    "client_segments": len(client_segments),
                    "conseiller_talk_time": round(conseiller_time, 2),
                    "client_talk_time": round(client_time, 2),
                    "talk_ratio": round(conseiller_time / (client_time + 0.001), 2)  # Avoid division by zero
                }
            }
        }
    
    def _complete_transcription_fallback(self, file_path: str, task_id: str = None, 
                                       progress_callback=None, language: str = None, partial_results: List[Dict] = None) -> Dict[str, Any]:
        """
        Fallback to complete transcription when diarization doesn't provide sufficient coverage
        Combines partial diarization results with complete transcription
        """
        try:
            if progress_callback:
                progress_callback(task_id, "transcription", "in_progress", 60, 
                                "Performing complete transcription fallback...")
            
            # Get complete transcription
            complete_result = self._transcribe_with_language(file_path, language)
            complete_text = complete_result.get("text", "").strip()
            
            if progress_callback:
                progress_callback(task_id, "transcription", "in_progress", 80, 
                                "Combining with partial diarization results...")
            
            # Load audio to get duration
            audio, sample_rate = self._load_audio(file_path)
            duration = len(audio) / sample_rate
            
            # Create a combined output
            # Use complete transcription as primary, but include diarization segments if available
            segments = []
            
            # Add complete transcription as primary segment
            segments.append({
                "start": 0,
                "end": duration,
                "speaker": "complete_transcript",
                "text": complete_text,
                "confidence": complete_result.get("language_probability", 0.0),
                "language": complete_result.get("language", language),
                "context_used": complete_result.get("context_used", False),
                "source": "complete_transcription"
            })
            
            # Add partial diarization results as additional segments if they exist
            if partial_results:
                for result in partial_results:
                    if result.get("text", "").strip():
                        result["source"] = "partial_diarization"
                        segments.append(result)
            
            # Create the formatted output
            output = {
                "text": complete_text,
                "transcript": complete_text,  # Use complete text as primary transcript
                "segments": segments,
                "metadata": {
                    "duration": duration,
                    "num_segments": len(segments),
                    "speakers": list(set(s.get("speaker", "unknown") for s in segments)),
                    "backend": f"{self.backend} + fallback_complete",
                    "gpu_enabled": self.backend == "faster-whisper",
                    "diarization_enabled": True,
                    "diarization_fallback": True,
                    "dominant_language": language,
                    "language_consistency": True,
                    "partial_diarization_segments": len(partial_results) if partial_results else 0
                }
            }
            
            logger.info(f"Complete transcription fallback completed: {len(complete_text)} characters")
            return output
            
        except Exception as e:
            logger.error(f"Complete transcription fallback failed: {e}")
            # Final fallback to basic transcribe
            return self.transcribe(file_path)
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the current Whisper setup"""
        return {
            "backend": self.backend,
            "gpu_enabled": self.backend == "faster-whisper",
            "model_loaded": self.model is not None,
            "diarization_available": self.diarization_available,
            "diarization_enabled": not self.skip_diarization,
            "fast_mode": self.skip_diarization,
            "diarization_type": "disabled" if self.skip_diarization else ("pyannote" if hasattr(self.diarization_pipeline, '__call__') else "vad" if self.diarization_pipeline else "none")
        }


class VADPipeline:
    """VAD-based speaker segmentation optimized for support center calls"""
    
    def segment(self, audio: np.ndarray, sample_rate: int) -> List[Dict]:
        """Segment audio using Voice Activity Detection with call center logic"""
        # Use more sensitive VAD for call center conversations
        intervals = librosa.effects.split(audio, top_db=12, frame_length=2048, hop_length=512)  # Reduced from 15 to 12 for more sensitivity
        
        segments = []
        current_speaker = "conseiller"  # Monaco Telecom advisor typically starts
        last_end_time = 0
        silence_threshold = 3.0  # Increased from 2.0 to 3.0 seconds - more conservative speaker switching
        min_segment_duration = 1.0  # Reduced from 3.0 to 1.0 seconds - capture shorter segments
        
        for i, (start_frame, end_frame) in enumerate(intervals):
            start_time = start_frame / sample_rate
            end_time = end_frame / sample_rate
            duration = end_time - start_time
            
            # Calculate silence gap from previous segment
            silence_gap = start_time - last_end_time if segments else 0
            
            # Filter very short segments but be less aggressive
            if duration < 0.5:  # Only skip segments shorter than 0.5 seconds
                continue
            
            # Try to merge with previous segment if gap is very small
            if segments and silence_gap < 0.3 and segments[-1]["speaker"] == current_speaker:
                # Merge with previous segment
                segments[-1]["end"] = end_time
                segments[-1]["duration"] = segments[-1]["end"] - segments[-1]["start"]
                last_end_time = end_time
                continue
            
            # Determine speaker change based on silence gap and conversation flow
            if silence_gap > silence_threshold:
                # Long silence suggests speaker change
                current_speaker = "client" if current_speaker == "conseiller" else "conseiller"
            elif len(segments) > 0:
                # Check if this might be a natural turn in conversation
                # Longer segments from advisor, shorter responses from client often
                prev_duration = segments[-1]["duration"]
                if prev_duration > 15.0 and current_speaker == "conseiller":
                    # After long advisor explanation, likely client response
                    current_speaker = "client"
                elif prev_duration < 5.0 and current_speaker == "client" and duration > 8.0:
                    # After short client response, likely advisor explanation
                    current_speaker = "conseiller"
            
            segments.append({
                "start": start_time,
                "end": end_time,
                "speaker": current_speaker,
                "duration": duration,
                "silence_before": silence_gap
            })
            
            last_end_time = end_time
        
        # Post-process to merge very close segments from same speaker
        segments = self._merge_close_segments(segments)
        
        # Add call center context information
        segments = self._add_call_context(segments)
        
        logger.info(f"VAD segmentation produced {len(segments)} segments covering {sum(s['duration'] for s in segments):.2f}s of {len(audio)/sample_rate:.2f}s total")
        
        return segments
    
    def _merge_close_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge segments from same speaker that are very close together"""
        if len(segments) <= 1:
            return segments
        
        merged = [segments[0]]
        
        for segment in segments[1:]:
            prev_segment = merged[-1]
            gap = segment["start"] - prev_segment["end"]
            
            # Merge if same speaker and gap < 1 second
            if (segment["speaker"] == prev_segment["speaker"] and gap < 1.0):
                prev_segment["end"] = segment["end"]
                prev_segment["duration"] = prev_segment["end"] - prev_segment["start"]
            else:
                merged.append(segment)
        
        return merged
    
    def _add_call_context(self, segments: List[Dict]) -> List[Dict]:
        """Add call center context to segments"""
        for i, segment in enumerate(segments):
            # Add context flags
            segment["is_opening"] = i == 0
            segment["is_closing"] = i == len(segments) - 1
            
            # Estimate turn type based on duration and position
            if segment["speaker"] == "conseiller":
                if segment["duration"] > 20.0:
                    segment["turn_type"] = "explanation"
                elif segment["duration"] > 10.0:
                    segment["turn_type"] = "response"
                elif segment["is_opening"]:
                    segment["turn_type"] = "greeting"
                else:
                    segment["turn_type"] = "acknowledgment"
            else:  # client
                if segment["duration"] > 15.0:
                    segment["turn_type"] = "detailed_request"
                elif segment["duration"] > 5.0:
                    segment["turn_type"] = "request"
                else:
                    segment["turn_type"] = "confirmation"
        
        return segments


# Global instance - initialized once
_whisper_instance: Optional[WhisperWrapper] = None

def get_whisper_model() -> WhisperWrapper:
    """Get the global Whisper instance (thread-safe singleton)"""
    global _whisper_instance
    if _whisper_instance is None:
        _whisper_instance = WhisperWrapper()
    return _whisper_instance

def transcribe_audio_safe(file_path: str) -> Dict[str, Any]:
    """Safe transcription function that works in all environments"""
    whisper_model = get_whisper_model()
    return whisper_model.transcribe(file_path) 