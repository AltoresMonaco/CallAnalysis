"""
Safe Whisper wrapper with GPU support and Speaker Diarization
Automatically falls back to CPU if GPU is not available
Includes offline-capable speaker diarization for support center calls
"""
import logging
import os
import tempfile
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

# Audio processing
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

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
        """Determine if GPU should be used based on environment and availability"""
        use_gpu_env = os.getenv('USE_GPU', 'false').lower()
        
        if use_gpu_env == 'false':
            return False
        elif use_gpu_env == 'true':
            return True
        elif use_gpu_env == 'auto':
            # Auto-detect: try GPU, fallback to CPU if not available
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                logger.info("PyTorch not available - using CPU mode")
                return False
        else:
            logger.warning(f"Unknown USE_GPU value: {use_gpu_env} - defaulting to CPU")
            return False
    
    def _initialize_model(self):
        """Initialize the appropriate Whisper model"""
        try:
            should_use_gpu = self._should_use_gpu()
            
            if should_use_gpu:
                logger.info("GPU mode requested - attempting to load faster-whisper")
                try:
                    from faster_whisper import WhisperModel
                    
                    # Try to initialize GPU model
                    self.model = WhisperModel(
                        "large-v3",
                        device="cuda",
                        compute_type="float16"
                    )
                    self.backend = "faster-whisper"
                    logger.info("✅ GPU-accelerated Whisper loaded successfully")
                    return
                    
                except Exception as gpu_error:
                    logger.warning(f"GPU Whisper failed: {gpu_error} - falling back to CPU")
            
            # Fallback to CPU mode (your current working setup)
            logger.info("Loading CPU-only Whisper (current setup)")
            import whisper
            self.model = whisper.load_model("base")
            self.backend = "openai-whisper"
            logger.info("✅ CPU Whisper loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            raise RuntimeError(f"Could not initialize Whisper model: {e}")
    
    def _initialize_diarization(self):
        """Initialize speaker diarization pipeline (offline-capable)"""
        try:
            logger.info("Initializing speaker diarization...")
            
            # Try to load pyannote diarization
            try:
                from pyannote.audio import Pipeline
                from pyannote.core import Segment, Annotation
                
                # Try to load a pre-trained diarization pipeline
                # This works offline if models are cached
                hf_token = os.getenv('HF_TOKEN')
                
                try:
                    if hf_token:
                        logger.info("Using HuggingFace token for pyannote authentication")
                        pipeline = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            use_auth_token=hf_token
                        )
                    else:
                        logger.info("No HF_TOKEN found, trying without authentication")
                        pipeline = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            use_auth_token=False
                        )
                    
                    if pipeline is not None:
                        self.diarization_pipeline = pipeline
                        self.diarization_available = True
                        logger.info("✅ Pyannote diarization loaded successfully")
                        return
                    else:
                        logger.warning("Pyannote pipeline returned None - falling back to VAD")
                        
                except Exception as auth_error:
                    logger.warning(f"Pyannote pipeline failed: {auth_error}")
                    if hf_token:
                        logger.info("💡 If you have a token, make sure to accept terms at: https://hf.co/pyannote/speaker-diarization-3.1")
                    else:
                        logger.info("💡 For better diarization, get token at: https://hf.co/settings/tokens")
                    # Try alternative approach or fallback
                    
            except ImportError:
                logger.info("pyannote.audio not available - using VAD fallback")
            
            # Fallback to VAD-based segmentation
            self.diarization_pipeline = self._create_vad_pipeline()
            self.diarization_available = True
            logger.info("✅ VAD-based speaker segmentation initialized")
            
        except Exception as e:
            logger.warning(f"Diarization initialization failed: {e}")
            self.diarization_pipeline = None
            self.diarization_available = False
            logger.info("Diarization disabled - will use basic transcription")
    
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
        """Perform speaker diarization"""
        if self.diarization_pipeline is None:
            raise RuntimeError("Diarization pipeline is not initialized")
        
        if hasattr(self.diarization_pipeline, '__call__'):
            # Pyannote pipeline
            return self._pyannote_diarization(file_path)
        else:
            # VAD pipeline
            return self.diarization_pipeline.segment(audio, sample_rate)
    
    def _pyannote_diarization(self, file_path: str) -> List[Dict]:
        """Use pyannote for diarization"""
        diarization = self.diarization_pipeline(file_path)
        
        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker,
                "duration": segment.duration
            })
        
        return segments
    
    def _transcribe_segments(self, file_path: str, segments: List[Dict], 
                           task_id: str, progress_callback=None, forced_language: str = None, context_prompt: str = None) -> List[Dict]:
        """Transcribe each speech segment with optional forced language and context prompt"""
        results = []
        total_segments = len(segments)
        
        for i, segment in enumerate(segments):
            try:
                # Extract segment audio
                segment_file = self._extract_segment(file_path, segment)
                
                # Transcribe segment with forced language and context
                result = self._transcribe_with_language(segment_file, forced_language, context_prompt)
                text = result.get("text", "").strip()
                
                # Clean up temp file
                os.unlink(segment_file)
                
                if text:
                    results.append({
                        "start": segment["start"],
                        "end": segment["end"], 
                        "speaker": segment["speaker"],
                        "text": text,
                        "confidence": result.get("language_probability", 0.0),
                        "language": result.get("language", forced_language),
                        "context_used": result.get("context_used", False)
                    })
                
                # Update progress
                if progress_callback:
                    progress = 30 + int((i + 1) / total_segments * 60)
                    progress_callback(task_id, "transcription", "in_progress", progress, 
                                    f"Transcribed segment {i+1}/{total_segments} ({forced_language})")
                
            except Exception as e:
                logger.warning(f"Failed to transcribe segment {i}: {e}")
                continue
        
        return results
    
    def _transcribe_with_language(self, file_path: str, language: str = None, context_prompt: str = None) -> Dict[str, Any]:
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
        for result in results:
            speaker = result["speaker"]
            text = result["text"]
            
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
                transcript_lines.append(f"{time_stamp} {label} ({turn_type}): {text}")
            else:
                transcript_lines.append(f"{time_stamp} {label}: {text}")
        
        transcript = "\n".join(transcript_lines)
        
        # Count conversation statistics
        conseiller_segments = [r for r in results if r["speaker"] == "conseiller"]
        client_segments = [r for r in results if r["speaker"] == "client"]
        
        conseiller_time = sum(r.get("end", 0) - r.get("start", 0) for r in conseiller_segments)
        client_time = sum(r.get("end", 0) - r.get("start", 0) for r in client_segments)
        
        return {
            "text": transcript,  # Keep compatible with basic transcribe
            "transcript": transcript,
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
        intervals = librosa.effects.split(audio, top_db=15, frame_length=2048, hop_length=512)
        
        segments = []
        current_speaker = "conseiller"  # Monaco Telecom advisor typically starts
        last_end_time = 0
        silence_threshold = 2.0  # 2 seconds of silence to change speaker
        min_segment_duration = 3.0  # Minimum 3 seconds for better context
        
        for i, (start_frame, end_frame) in enumerate(intervals):
            start_time = start_frame / sample_rate
            end_time = end_frame / sample_rate
            duration = end_time - start_time
            
            # Calculate silence gap from previous segment
            silence_gap = start_time - last_end_time if segments else 0
            
            # Filter very short segments (merge them)
            if duration < min_segment_duration:
                # Try to merge with previous segment if gap is small
                if segments and silence_gap < 1.0:
                    segments[-1]["end"] = end_time
                    segments[-1]["duration"] = segments[-1]["end"] - segments[-1]["start"]
                    last_end_time = end_time
                    continue
                elif duration < 1.5:  # Skip very short segments
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