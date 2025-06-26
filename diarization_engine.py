"""
Diarisation engine 100% open source avec pyannote
Optimisé par Altores pour support téléphonique Monaco Telecom
Compatible avec RunPod PyTorch 2.4.0
"""

import logging
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import torch de façon sécurisée
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Pyannote imports avec gestion d'erreur
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment, Annotation
    PYANNOTE_AVAILABLE = True
except ImportError as e:
    PYANNOTE_AVAILABLE = False
    pyannote_error = str(e)

# Audio processing de base
try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

logger = logging.getLogger(__name__)

class PyannoteEngine:
    """Engine diarisation pyannote open source - Version compatible PyTorch 2.4.0"""
    
    def __init__(self):
        self.device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.pipeline = None
        self.is_loaded = False
        
        # Config support téléphonique
        self.call_center_config = {
            "min_speakers": 2,
            "max_speakers": 3,  # Agent + Client + (système)
            "min_segment_duration": 1.0,
            "min_speaker_duration": 2.0
        }
        
        # Log des disponibilités
        logger.info(f"PyannoteEngine init - TORCH: {TORCH_AVAILABLE}, PYANNOTE: {PYANNOTE_AVAILABLE}, AUDIO: {AUDIO_AVAILABLE}")
        if not PYANNOTE_AVAILABLE:
            logger.warning(f"Pyannote not available: {pyannote_error if 'pyannote_error' in locals() else 'Unknown error'}")
    
    def load_models(self) -> bool:
        """Charge les modèles pyannote open source avec gestion d'erreur améliorée"""
        if not PYANNOTE_AVAILABLE:
            logger.error("Pyannote not available - cannot load models")
            return False
            
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available - cannot load models")
            return False
            
        try:
            logger.info(f"Loading pyannote models on {self.device}...")
            
            # Essayer différents modèles open source avec gestion d'erreur par modèle
            model_options = [
                "pyannote/speaker-diarization-3.1",
                "pyannote/speaker-diarization",
                "pyannote/segmentation-3.0"
            ]
            
            for model_name in model_options:
                try:
                    logger.info(f"Trying model: {model_name}")
                    
                    # Chargement avec paramètres simplifiés pour éviter les conflits
                    self.pipeline = Pipeline.from_pretrained(
                        model_name,
                        use_auth_token=False  # Pas de token
                    )
                    
                    # Configurer pour GPU si disponible (avec gestion d'erreur)
                    if self.device == "cuda" and torch.cuda.is_available():
                        try:
                            self.pipeline = self.pipeline.to(torch.device("cuda"))
                            logger.info(f"Model moved to GPU successfully")
                        except Exception as gpu_error:
                            logger.warning(f"Failed to move model to GPU: {gpu_error}, using CPU")
                            self.device = "cpu"
                    
                    self.is_loaded = True
                    logger.info(f"✅ Pyannote loaded: {model_name} on {self.device}")
                    return True
                    
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            logger.error("❌ No pyannote models could be loaded")
            return False
            
        except Exception as e:
            logger.error(f"Pyannote initialization failed: {e}")
            return False
    
    def diarize_audio(self, audio_path: str) -> List[Dict]:
        """
        Diarise un fichier audio avec optimisations téléphonie
        """
        if not self.is_loaded:
            if not self.load_models():
                raise RuntimeError("Pyannote models not available")
        
        try:
            logger.info(f"Starting pyannote diarization: {audio_path}")
            
            # Lancer diarisation avec gestion d'erreur
            diarization = self.pipeline(audio_path)
            
            # Convertir en format standard
            segments = self._convert_to_segments(diarization)
            
            # Post-processing support téléphonique
            segments = self._optimize_for_call_center(segments)
            
            logger.info(f"Pyannote diarization completed: {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"Pyannote diarization failed: {e}")
            raise
    
    def _convert_to_segments(self, diarization: Annotation) -> List[Dict]:
        """Convertit pyannote annotation en format segments"""
        segments = []
        
        try:
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker,
                    "duration": segment.duration
                })
        except Exception as e:
            logger.error(f"Error converting segments: {e}")
            raise
        
        return segments
    
    def _optimize_for_call_center(self, segments: List[Dict]) -> List[Dict]:
        """
        Post-processing spécifique support téléphonique
        """
        if not segments:
            return segments
        
        try:
            # 1. Filtrer segments trop courts
            filtered_segments = [
                s for s in segments 
                if s["duration"] >= self.call_center_config["min_segment_duration"]
            ]
            
            # 2. Mapper speakers → rôles téléphoniques
            optimized_segments = self._map_to_call_roles(filtered_segments)
            
            # 3. Fusionner segments proches du même speaker
            merged_segments = self._merge_close_segments(optimized_segments)
            
            return merged_segments
            
        except Exception as e:
            logger.error(f"Error in call center optimization: {e}")
            return segments  # Retourner segments originaux en cas d'erreur
    
    def _map_to_call_roles(self, segments: List[Dict]) -> List[Dict]:
        """
        Mappe les speakers pyannote aux rôles support téléphonique
        """
        # Analyser qui parle en premier (généralement l'agent)
        if not segments:
            return segments
        
        try:
            # Trier par temps de début
            sorted_segments = sorted(segments, key=lambda x: x["start"])
            
            # Le premier speaker est généralement l'agent Monaco Telecom
            first_speaker = sorted_segments[0]["speaker"]
            
            # Mapping speakers → rôles
            speaker_mapping = {}
            unique_speakers = list(set(s["speaker"] for s in segments))
            
            if len(unique_speakers) >= 2:
                speaker_mapping[first_speaker] = "conseiller"
                # Le deuxième speaker est le client
                other_speakers = [s for s in unique_speakers if s != first_speaker]
                speaker_mapping[other_speakers[0]] = "client"
                
                # Speakers supplémentaires = système/attente
                for i, speaker in enumerate(other_speakers[1:], 2):
                    speaker_mapping[speaker] = f"speaker_{i+1}"
            else:
                # Un seul speaker détecté
                speaker_mapping[unique_speakers[0]] = "conseiller"
            
            # Appliquer le mapping
            for segment in segments:
                segment["speaker"] = speaker_mapping.get(
                    segment["speaker"], 
                    segment["speaker"]
                )
            
            return segments
            
        except Exception as e:
            logger.error(f"Error in speaker mapping: {e}")
            return segments
    
    def _merge_close_segments(self, segments: List[Dict]) -> List[Dict]:
        """Fusionne segments proches du même speaker"""
        if len(segments) <= 1:
            return segments
        
        try:
            merged = [segments[0]]
            
            for segment in segments[1:]:
                prev_segment = merged[-1]
                gap = segment["start"] - prev_segment["end"]
                
                # Fusionner si même speaker et gap < 2 secondes
                if (segment["speaker"] == prev_segment["speaker"] and gap < 2.0):
                    prev_segment["end"] = segment["end"]
                    prev_segment["duration"] = prev_segment["end"] - prev_segment["start"]
                else:
                    merged.append(segment)
            
            return merged
            
        except Exception as e:
            logger.error(f"Error in segment merging: {e}")
            return segments

# Instance globale
_pyannote_engine = None

def get_pyannote_engine() -> PyannoteEngine:
    """Singleton pyannote engine avec gestion d'erreur"""
    global _pyannote_engine
    if _pyannote_engine is None:
        _pyannote_engine = PyannoteEngine()
    return _pyannote_engine 