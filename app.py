# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from whisper_wrapper import get_whisper_model
import tempfile
import os
import json
import requests
from pathlib import Path
import uvicorn
import logging
import traceback
from typing import Optional, Dict, Any
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Transcript Analysis API", version="1.0.0")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add a thread pool executor for background processing
executor = ThreadPoolExecutor(max_workers=2)

# Create results directory for persistence
RESULTS_DIR = Path("saved_results")
RESULTS_DIR.mkdir(exist_ok=True)

# ---- Dynamic Steps and Prompts Configuration ----
class StepConfig(BaseModel):
    step_id: str
    name: str
    step_type: str = "llm"  # "audio" or "llm"
    prompt_file: Optional[str] = None  # Only for LLM steps
    enabled: bool = True
    order: int
    description: Optional[str] = None
    result_key: str  # Key to store result in task_state.results
    input_source: str = "transcript"  # "transcript" or result_key of another step (e.g., "summary")
    parameters: Optional[Dict[str, Any]] = None  # Step-specific parameters

class AudioStepConfig(BaseModel):
    step_id: str
    name: str
    step_type: str = "audio"
    enabled: bool = True
    order: int
    description: Optional[str] = None
    result_key: str
    parameters: Dict[str, Any]  # Audio processing parameters

class PromptData(BaseModel):
    filename: str
    content: str
    description: Optional[str] = None

class StepsConfiguration(BaseModel):
    steps: Dict[str, StepConfig]

# Default audio processing steps
DEFAULT_AUDIO_STEPS = {
    "audio_loading": StepConfig(
        step_id="audio_loading",
        name="Audio Loading & Preprocessing",
        step_type="audio",
        enabled=True,
        order=1,
        description="Load and preprocess audio file",
        result_key="audio_metadata",
        input_source="file",
        parameters={
            "sample_rate": 16000,
            "mono": True,
            "normalize": True,
            "trim_silence": False,
            "trim_threshold_db": 20
        }
    ),
    "language_detection": StepConfig(
        step_id="language_detection",
        name="Language Detection",
        step_type="audio",
        enabled=True,
        order=2,
        description="Detect dominant language in audio",
        result_key="language_info",
        input_source="audio_metadata",
        parameters={
            "detection_method": "sample",  # "sample" or "full"
            "sample_duration": 30,
            "sample_position": "middle",  # "start", "middle", "end"
            "default_language": "fr",
            "confidence_threshold": 0.7
        }
    ),
    "diarization": StepConfig(
        step_id="diarization",
        name="Speaker Diarization",
        step_type="audio",
        enabled=True,
        order=3,
        description="Identify and segment speakers",
        result_key="diarization_segments",
        input_source="audio_metadata",
        parameters={
            "method": "vad",  # "pyannote", "vad", "disabled"
            "min_segment_duration": 1.0,
            "max_segment_duration": 30.0,
            "silence_threshold": 3.0,
            "merge_threshold": 1.0,
            "vad_sensitivity": 12,  # top_db for VAD
            "speaker_labels": {
                "default_first": "conseiller",
                "default_second": "client"
            },
            "max_speakers": 3,
            # Enhanced pyannote speaker context parameters
            "domain": "call_center",
            "scenario": "customer_support", 
            "expected_speakers": ["agent", "customer"],
            "speaker_roles": {
                "agent": ["conseiller", "support", "agent"],
                "customer": ["client", "customer", "caller"]
            },
            "conversation_type": "telephone",
            "language": "french",
            "context_prompt": "Call center conversation between Monaco Telecom support agent and customer",
            "min_speakers": 2,
            "speaker_change_threshold": 0.5,
            "min_speaker_duration": 2.0
        }
    ),
    "transcription": StepConfig(
        step_id="transcription",
        name="Audio Transcription",
        step_type="audio",
        enabled=True,
        order=4,
        description="Convert speech to text",
        result_key="transcript",
        input_source="diarization_segments",
        parameters={
            "backend": "auto",  # "faster-whisper", "openai-whisper", "auto"
            "model_size": "large-v3",
            "use_gpu": True,
            "compute_type": "float16",
            "chunk_processing": True,
            "chunk_size": 30,
            "overlap": 5,
            "force_language": None,  # None = use detected language
            "initial_prompt": None,
            "temperature": 0.0,
            "best_of": 1,
            "beam_size": 5
        }
    )
}

# Default LLM analysis steps (updated structure)
DEFAULT_LLM_STEPS = {
    "qualification": StepConfig(
        step_id="qualification",
        name="Qualification Analysis",
        step_type="llm",
        prompt_file="qualification.txt",
        enabled=True,
        order=5,
        description="Analyzes call qualification and satisfaction",
        result_key="summary",
        input_source="transcript"
    ),
    "critical_issues": StepConfig(
        step_id="critical_issues",
        name="Critical Issues Detection",
        step_type="llm",
        prompt_file="keywords_extract.txt",
        enabled=True,
        order=6,
        description="Identifies critical issues and concerns",
        result_key="issues",
        input_source="transcript"
    ),
    "categorisation": StepConfig(
        step_id="categorisation",
        name="Call Categorization",
        step_type="llm",
        prompt_file="categorise.txt",
        enabled=True,
        order=7,
        description="Categorizes the call based on commercial and technical typologies",
        result_key="categorisation",
        input_source="summary"  # Use summary from qualification step
    )
}

# Combine all default steps
DEFAULT_STEPS = {**DEFAULT_AUDIO_STEPS, **DEFAULT_LLM_STEPS}

# Load or initialize steps configuration
def load_steps_config() -> Dict[str, StepConfig]:
    """Load steps configuration from file"""
    config_file = "settings/steps_config.json"
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                steps = {}
                for step_id, step_data in config_data.get("steps", {}).items():
                    steps[step_id] = StepConfig(**step_data)
                logger.info(f"Loaded steps configuration: {list(steps.keys())}")
                return steps
        except Exception as e:
            logger.error(f"Error loading steps config: {e}")
    
    logger.info("Using default steps configuration")
    return DEFAULT_STEPS.copy()

def save_steps_config(steps: Dict[str, StepConfig]):
    """Save steps configuration to file"""
    config_file = "settings/steps_config.json"
    os.makedirs("settings", exist_ok=True)
    
    try:
        config_data = {
            "steps": {step_id: step.model_dump() for step_id, step in steps.items()}
        }
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Steps configuration saved: {list(steps.keys())}")
    except Exception as e:
        logger.error(f"Error saving steps config: {e}")
        raise

# Load initial steps configuration
STEPS_CONFIG = load_steps_config()

@app.get("/")
async def serve_index():
    """Serve the main reporting dashboard as the index page"""
    return FileResponse("static/reporting.html")

@app.get("/upload")
async def serve_upload_page():
    """Serve the upload/analysis page"""
    return FileResponse("static/upload.html")

@app.get("/ide")
async def serve_ide():
    """Serve the IDE interface"""
    return FileResponse("static/ide.html")

@app.get("/browser")
async def serve_browser():
    """Serve the browser interface"""
    return FileResponse("static/browser.html")

# File system API for IDE
@app.get("/api/files")
async def list_files(path: str = "."):
    """List files and directories"""
    try:
        items = []
        full_path = Path(path).resolve()
        
        # Security check - don't allow going outside current directory
        if not str(full_path).startswith(str(Path.cwd())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if full_path.is_dir():
            for item in sorted(full_path.iterdir()):
                items.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "path": str(item.relative_to(Path.cwd())),
                    "size": item.stat().st_size if item.is_file() else 0,
                    "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                })
        
        return {"items": items, "path": str(full_path.relative_to(Path.cwd()))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files/content")
async def get_file_content(path: str):
    """Get file content"""
    try:
        full_path = Path(path).resolve()
        
        # Security check
        if not str(full_path).startswith(str(Path.cwd())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not full_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        
        async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        return {"content": content, "path": path}
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is binary or not UTF-8 encoded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class FileContent(BaseModel):
    content: str

@app.post("/api/files/content")
async def save_file_content(path: str, file_data: FileContent):
    """Save file content"""
    try:
        # Handle both absolute and relative paths
        if path.startswith('./') or not path.startswith('/'):
            full_path = Path(path).resolve()
        else:
            full_path = Path.cwd() / path.lstrip('/')
        
        # Security check
        if not str(full_path).startswith(str(Path.cwd())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Create directory if it doesn't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(full_path, 'w', encoding='utf-8') as f:
            await f.write(file_data.content)
        
        return {"message": "File saved successfully", "path": str(full_path.relative_to(Path.cwd()))}
    except Exception as e:
        logger.error(f"Error saving file {path}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/files")
async def delete_file(path: str):
    """Delete a file or directory"""
    try:
        full_path = Path(path).resolve()
        
        # Security check
        if not str(full_path).startswith(str(Path.cwd())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if full_path.is_file():
            full_path.unlink()
        elif full_path.is_dir():
            import shutil
            shutil.rmtree(full_path)
        else:
            raise HTTPException(status_code=404, detail="File or directory not found")
        
        return {"message": "Deleted successfully", "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- Settings Management ----
class Settings(BaseModel):
    ollama_url: str
    ollama_model: str

class ProgressUpdate(BaseModel):
    task_id: str
    step: str
    status: str  # 'in_progress', 'completed', 'error'
    message: str
    progress: int  # 0-100
    details: Optional[Dict[str, Any]] = None

# ---- Load/Save Settings Functions ----
def load_settings() -> dict:
    """Load settings from file"""
    settings_file = "settings/config.json"
    
    # Create settings directory if it doesn't exist
    os.makedirs("settings", exist_ok=True)
    
    # Default settings
    defaults = {
        "ollama_url": os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/api/generate"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3")
    }
    
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r') as f:
                saved_settings = json.load(f)
                logger.info(f"Loaded settings from {settings_file}: {saved_settings}")
                return saved_settings
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
    
    logger.info(f"Using default settings: {defaults}")
    return defaults

def save_settings(settings: dict):
    """Save settings to file"""
    settings_file = "settings/config.json"
    os.makedirs("settings", exist_ok=True)
    
    try:
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        logger.info(f"Settings saved to {settings_file}: {settings}")
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        raise

# Load initial settings
current_settings = load_settings()
OLLAMA_URL = current_settings["ollama_url"]
OLLAMA_MODEL = current_settings["ollama_model"]

logger.info(f"Starting with Ollama URL: {OLLAMA_URL}, Model: {OLLAMA_MODEL}")

# Load Whisper model once at startup using wrapper
logger.info("Loading Whisper model...")
try:
    WHISPER_MODEL = get_whisper_model()
    whisper_info = WHISPER_MODEL.get_info()
    logger.info(f"Whisper model loaded successfully: {whisper_info['backend']} (GPU: {whisper_info['gpu_enabled']})")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    WHISPER_MODEL = None

# ---- Load Prompts from Files ----
def load_prompt(filename: str) -> str:
    """Load prompt content from file"""
    prompt_path = Path("prompts") / filename
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            logger.info(f"Loaded prompt from {filename} ({len(content)} characters)")
            return content
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_path}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prompt file not found: {prompt_path}"
        )
    except Exception as e:
        logger.error(f"Error loading prompt {prompt_path}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error loading prompt file {prompt_path}: {str(e)}"
        )

# Prompts are now loaded dynamically from the step configuration

# ---- Functions ----

class TaskState:
    """Modern task state management with dynamic steps"""
    def __init__(self, task_id: str, filename: str, include_transcription: bool = True):
        self.task_id = task_id
        self.filename = filename
        self.status = "initializing"  # initializing, processing, completed, failed
        self.current_step = None
        
        # Initialize steps based on current configuration
        self.steps = {}
        
        # Always include transcription if requested
        if include_transcription:
            self.steps["transcription"] = {
                "status": "pending", 
                "progress": 0, 
                "message": "Waiting to start...", 
                "details": {}
            }
        
        # Add enabled steps from configuration
        enabled_steps = {k: v for k, v in STEPS_CONFIG.items() if v.enabled}
        sorted_steps = sorted(enabled_steps.items(), key=lambda x: x[1].order)
        
        for step_id, step_config in sorted_steps:
            self.steps[step_id] = {
                "status": "pending", 
                "progress": 0, 
                "message": "Waiting to start...", 
                "details": {}
            }
        
        # Initialize results dict with keys from step configs
        self.results = {"transcript": None}
        for step_id, step_config in STEPS_CONFIG.items():
            if step_config.enabled:
                self.results[step_config.result_key] = None
        
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.error_message = None

    def update_step(self, step_name: str, status: str, progress: int, message: str, details: dict = None):
        """Update a specific step's status"""
        if step_name in self.steps:
            self.steps[step_name].update({
                "status": status,
                "progress": progress,
                "message": message,
                "details": details or {}
            })
            self.current_step = step_name
            self.updated_at = datetime.now().isoformat()
            
            # Log the update
            logger.info(f"[{self.task_id}] Step '{step_name}' updated: {status} ({progress}%) - {message}")
            
            # Update overall task status
            if status == "error":
                self.status = "failed"
                self.error_message = message
            elif status == "completed" and step_name == "transcription":
                self.status = "processing"
            elif status == "completed" and all(
                step["status"] in ["completed", "error"] for step in self.steps.values()
            ):
                self.status = "completed"

    def get_state(self):
        """Get the complete task state for API responses"""
        return {
            "task_id": self.task_id,
            "filename": self.filename,
            "status": self.status,
            "current_step": self.current_step,
            "steps": self.steps,
            "results": self.results,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error_message": self.error_message
        }

# Replace the global progress store with a proper task store
task_store: Dict[str, TaskState] = {}

def get_task_state(task_id: str) -> TaskState:
    """Get or create a task state"""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_store[task_id]

def create_task_state(task_id: str, filename: str, include_transcription: bool = True) -> TaskState:
    """Create a new task state with dynamic steps"""
    task_state = TaskState(task_id, filename, include_transcription)
    task_store[task_id] = task_state
    enabled_steps = [step_id for step_id, config in STEPS_CONFIG.items() if config.enabled]
    logger.info(f"[{task_id}] Created new task state for file: {filename}, enabled steps: {enabled_steps}")
    return task_state

# Remove the old update_progress function and replace with modern approach
def update_task_step(task_id: str, step_name: str, status: str, progress: int, message: str, details: dict = None):
    """Update a task step with proper state management"""
    try:
        task_state = get_task_state(task_id)
        task_state.update_step(step_name, status, progress, message, details)
    except Exception as e:
        logger.error(f"[{task_id}] Failed to update step {step_name}: {e}")

def transcribe_audio(file_path: str, task_id: str):
    """Transcribe audio file using enhanced Whisper wrapper with speaker diarization"""
    try:
        # Check if model is available
        if WHISPER_MODEL is None:
            error_msg = "Whisper model not loaded. Please restart the server."
            update_task_step(task_id, "transcription", "error", 0, error_msg)
            logger.error(f"[{task_id}] {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        update_task_step(task_id, "transcription", "in_progress", 10, "Starting enhanced transcription...")
        
        # Try enhanced transcription with diarization first
        try:
            result = WHISPER_MODEL.transcribe_with_diarization(
                file_path, 
                task_id, 
                progress_callback=update_task_step
            )
            
            # Extract transcript and metadata
            transcript = result.get("text", result.get("transcript", ""))
            metadata = result.get("metadata", {})
            segments = result.get("segments", [])
            
            # Store result in task state
            task_state = get_task_state(task_id)
            task_state.results["transcript"] = transcript
            if segments:  # Only store if we have diarization segments
                task_state.results["diarization_segments"] = segments
            
            logger.info(f"[{task_id}] Enhanced transcription completed: {len(transcript)} characters, {len(segments)} segments")
            return transcript
            
        except Exception as e:
            logger.warning(f"[{task_id}] Enhanced transcription failed ({e}), falling back to basic transcription")
            return _fallback_transcription(file_path, task_id)
        
    except Exception as e:
        error_msg = f"Transcription failed: {str(e)}"
        update_task_step(task_id, "transcription", "error", 0, error_msg)
        logger.error(f"[{task_id}] {error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

def _fallback_transcription(file_path: str, task_id: str):
    """Fallback to basic Whisper transcription without diarization"""
    update_task_step(task_id, "transcription", "in_progress", 30, "Using basic transcription mode...")
    
    whisper_info = WHISPER_MODEL.get_info()
    backend_info = f"Using {whisper_info['backend']} ({'GPU' if whisper_info['gpu_enabled'] else 'CPU'})"
    
    update_task_step(task_id, "transcription", "in_progress", 50, f"Processing audio file... {backend_info}")
    
    # Use the wrapper's transcribe method
    result = WHISPER_MODEL.transcribe(file_path)
    transcript = result["text"]
    
    # Store result in task state
    task_state = get_task_state(task_id)
    task_state.results["transcript"] = transcript
    
    update_task_step(task_id, "transcription", "completed", 100, 
                    f"Basic transcription completed ({len(transcript)} characters)", {
                        "transcript_length": len(transcript),
                        "language": result.get("language", "unknown"),
                        "backend": result.get("backend", "unknown"),
                        "gpu_enabled": whisper_info['gpu_enabled'],
                        "mode": "basic"
                    })
    
    logger.info(f"[{task_id}] Basic transcription completed: {len(transcript)} characters using {result.get('backend', 'unknown')}")
    return transcript

def query_ollama(prompt: str, transcript: str, task_id: str, step_name: str):
    """Query Ollama for analysis with modern progress tracking"""
    global OLLAMA_URL, OLLAMA_MODEL
    
    try:
        update_task_step(task_id, step_name, "in_progress", 10, f"Preparing {step_name} analysis...")
        
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{prompt.strip()}\n\n{transcript.strip()}",
            "stream": False,
            "keep_alive": -1  # Keep model cached indefinitely
        }
        
        logger.info(f"[{task_id}] Starting Ollama query for {step_name}")
        
        update_task_step(task_id, step_name, "in_progress", 30, f"Sending request to Ollama ({OLLAMA_MODEL})...")
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        
        logger.info(f"[{task_id}] Ollama response for {step_name}: {response.status_code}")
        
        if response.status_code != 200:
            error_msg = f"Ollama returned status {response.status_code}: {response.text}"
            update_task_step(task_id, step_name, "error", 0, error_msg)
            logger.error(f"[{task_id}] {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        update_task_step(task_id, step_name, "in_progress", 70, "Processing Ollama response...")
        
        result = response.json()
        response_text = result.get("response", "")
        
        if not response_text:
            error_msg = f"Empty response from Ollama for {step_name}"
            update_task_step(task_id, step_name, "error", 0, error_msg)
            logger.error(f"[{task_id}] {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        update_task_step(task_id, step_name, "in_progress", 90, "Parsing response...")
        
        # Parse the response
        parsed_result = parse_json_response(response_text, task_id, step_name)
        
        # Store result in task state using dynamic step configuration
        task_state = get_task_state(task_id)
        if step_name in STEPS_CONFIG:
            result_key = STEPS_CONFIG[step_name].result_key
            task_state.results[result_key] = parsed_result
        else:
            # Fallback for custom steps
            task_state.results[step_name] = parsed_result
        
        update_task_step(task_id, step_name, "completed", 100, 
                        f"{step_name.title()} analysis completed", {
                            "response_length": len(response_text),
                            "model_used": OLLAMA_MODEL
                        })
        
        logger.info(f"[{task_id}] {step_name} completed successfully")
        return parsed_result
        
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Cannot connect to Ollama: {str(e)}"
        update_task_step(task_id, step_name, "error", 0, error_msg)
        logger.error(f"[{task_id}] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    except requests.exceptions.Timeout as e:
        error_msg = f"Ollama request timed out: {str(e)}"
        update_task_step(task_id, step_name, "error", 0, error_msg)
        logger.error(f"[{task_id}] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"{step_name} analysis failed: {str(e)}"
        update_task_step(task_id, step_name, "error", 0, error_msg)
        logger.error(f"[{task_id}] {error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

def parse_json_response(response_text: str, task_id: str, step_name: str):
    """Parse JSON response from Ollama with modern error handling"""
    try:
        # Try direct JSON parsing first
        try:
            result = json.loads(response_text)
            logger.info(f"[{task_id}] {step_name} JSON parsed successfully")
            return result
        except json.JSONDecodeError:
            logger.warning(f"[{task_id}] Direct JSON parsing failed for {step_name}, trying extraction...")
        
        # Try to extract JSON from response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        
        if start != -1 and end > start:
            json_part = response_text[start:end]
            try:
                result = json.loads(json_part)
                logger.info(f"[{task_id}] {step_name} JSON extracted and parsed")
                return result
            except json.JSONDecodeError:
                pass
        
        # Convert text to structured format
        if step_name == "critical_issues":
            issues = []
            current_issue = {}
            
            for line in response_text.split('\n'):
                line = line.strip()
                if not line:
                    if current_issue:
                        issues.append(current_issue)
                        current_issue = {}
                    continue
                    
                if line.startswith('Issue:'):
                    if current_issue:
                        issues.append(current_issue)
                    current_issue = {'issue': line.replace('Issue:', '').strip()}
                elif line.startswith('Relevance:'):
                    current_issue['relevance'] = line.replace('Relevance:', '').strip()
                elif line.startswith('Related keywords:'):
                    current_issue['keywords'] = line.replace('Related keywords:', '').strip()
            
            if current_issue:
                issues.append(current_issue)
            
            return {'issues': issues} if issues else {'issues': []}
        
        elif step_name == "qualification":
            qualification = {}
            for line in response_text.split('\n'):
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    qualification[key] = value
            return qualification
        
        # Fallback: return as text
        return {"response": response_text}
        
    except Exception as e:
        error_msg = f"Failed to parse {step_name} response: {str(e)}"
        logger.error(f"[{task_id}] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

# ---- API Endpoints ----

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    whisper_info = WHISPER_MODEL.get_info() if WHISPER_MODEL else None
    return {
        "message": "Audio Transcript Analysis API is running", 
        "timestamp": datetime.now().isoformat(),
        "whisper_model_loaded": WHISPER_MODEL is not None,
        "whisper_backend": whisper_info.get("backend") if whisper_info else None,
        "gpu_enabled": whisper_info.get("gpu_enabled") if whisper_info else None
    }

@app.get("/results/{task_id}")
async def get_results(task_id: str):
    """Get final results for a completed task"""
    try:
        task_state = get_task_state(task_id)
        
        if task_state.status == "initializing":
            return {"status": "initializing", "message": "Task is being initialized"}
        elif task_state.status == "processing":
            return {"status": "processing", "message": "Task is still processing"}
        elif task_state.status in ["completed", "failed"]:
            return {
                "status": task_state.status,
                "task_id": task_state.task_id,
                "filename": task_state.filename,
                **task_state.results
            }
        else:
            return {"status": "unknown", "message": "Task status unknown"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting results for {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get results")

@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """Get progress for a specific task"""
    try:
        task_state = get_task_state(task_id)
        return task_state.get_state()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting progress for {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get progress")

@app.get("/settings")
async def get_settings():
    """Get current settings"""
    return {
        "ollama_url": OLLAMA_URL,
        "ollama_model": OLLAMA_MODEL
    }

@app.post("/settings")
async def update_settings(settings: Settings):
    """Update settings"""
    global OLLAMA_URL, OLLAMA_MODEL
    
    # Validate URL format
    if not settings.ollama_url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Ollama URL must start with http:// or https://")
    
    # Test connection to new Ollama URL
    test_url = settings.ollama_url
    if not test_url.endswith('/api/generate'):
        if test_url.endswith('/'):
            test_url += 'api/generate'
        else:
            test_url += '/api/generate'
    
    try:
        # Test with a simple request
        test_payload = {
            "model": settings.ollama_model,
            "prompt": "Hello",
            "stream": False
        }
        response = requests.post(test_url, json=test_payload, timeout=10)
        if response.status_code == 404:
            # Model might not exist, but connection is working
            pass
        elif response.status_code >= 500:
            raise HTTPException(status_code=400, detail="Ollama server error - check if the service is running")
        
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=400, detail="Cannot connect to Ollama at the specified URL")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=400, detail="Connection to Ollama timed out")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to connect to Ollama: {str(e)}")
    
    # Update global settings
    OLLAMA_URL = test_url
    OLLAMA_MODEL = settings.ollama_model
    
    # Save to file
    save_settings({
        "ollama_url": OLLAMA_URL,
        "ollama_model": OLLAMA_MODEL
    })
    
    return {"message": "Settings updated successfully", "ollama_url": OLLAMA_URL, "ollama_model": OLLAMA_MODEL}

@app.post("/test-ollama")
async def test_ollama_connection():
    """Test current Ollama connection and model"""
    global OLLAMA_URL, OLLAMA_MODEL
    
    try:
        # Test connection with current settings
        test_payload = {
            "model": OLLAMA_MODEL,
            "prompt": "Hello, this is a test. Please respond with 'Connection successful'.",
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=test_payload, timeout=15)
        response.raise_for_status()
        
        result = response.json()
        return {
            "status": "success",
            "message": "Ollama connection successful",
            "model": OLLAMA_MODEL,
            "url": OLLAMA_URL,
            "test_response": result.get("response", "No response")[:100] + "..." if len(result.get("response", "")) > 100 else result.get("response", "")
        }
        
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "message": "Cannot connect to Ollama - check if the service is running",
            "url": OLLAMA_URL
        }
    except requests.exceptions.Timeout:
        return {
            "status": "error", 
            "message": "Connection to Ollama timed out",
            "url": OLLAMA_URL
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Ollama test failed: {str(e)}",
            "url": OLLAMA_URL
        }

@app.get("/prompts")
async def get_prompts():
    """Get current prompts for debugging/verification"""
    prompts = {}
    try:
        # Load prompts dynamically from current step configuration
        for step_id, step_config in STEPS_CONFIG.items():
            if step_config.enabled:
                try:
                    prompt_content = load_prompt(step_config.prompt_file)
                    preview = prompt_content[:200] + "..." if len(prompt_content) > 200 else prompt_content
                    prompts[step_id] = {
                        "file": step_config.prompt_file,
                        "preview": preview,
                        "full_length": len(prompt_content)
                    }
                except Exception as e:
                    prompts[step_id] = {
                        "file": step_config.prompt_file,
                        "error": f"Failed to load: {str(e)}"
                    }
        return prompts
    except Exception as e:
        return {"error": f"Failed to load prompts: {str(e)}"}

@app.post("/reload-prompts")
async def reload_prompts():
    """Reload prompts from files without restarting the server"""
    try:
        # Test loading all prompts from current step configuration
        loaded_prompts = []
        errors = []
        
        for step_id, step_config in STEPS_CONFIG.items():
            if step_config.enabled:
                try:
                    load_prompt(step_config.prompt_file)  # Test load
                    loaded_prompts.append(f"{step_id} ({step_config.prompt_file})")
                except Exception as e:
                    errors.append(f"{step_id} ({step_config.prompt_file}): {str(e)}")
        
        if errors:
            return {
                "message": "Some prompts failed to load",
                "loaded": loaded_prompts,
                "errors": errors
            }
        else:
            return {
                "message": "All prompts validated successfully",
                "loaded": loaded_prompts
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload prompts: {str(e)}")

@app.get("/results")
async def serve_results_page():
    """Serve the results viewing page"""
    try:
        with open("static/results.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Results page not found")

@app.get("/manage")
async def serve_manage_page():
    """Serve the steps and prompts management page"""
    try:
        with open("static/manage.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Management page not found")

@app.get("/agent")
async def serve_agent_page():
    """Serve the agent dashboard page"""
    try:
        with open("static/agentpage.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Agent page not found")

@app.get("/reporting")
async def serve_reporting_page():
    """Serve the customer care intelligence dashboard page"""
    try:
        with open("static/reporting.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Reporting page not found")

@app.get("/api/saved-results")
async def get_all_saved_results():
    """Get all saved results for the results page"""
    try:
        saved_results = load_saved_results()
        return {"results": saved_results, "count": len(saved_results)}
    except Exception as e:
        logger.error(f"Error loading saved results: {e}")
        raise HTTPException(status_code=500, detail="Failed to load saved results")

@app.get("/api/saved-results/{task_id}")
async def get_saved_result_by_id(task_id: str):
    """Get a specific saved result by task ID"""
    try:
        result = get_saved_result(task_id)
        if result:
            return result
        else:
            raise HTTPException(status_code=404, detail="Result not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading result {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load result")

@app.put("/api/saved-results/{task_id}")
async def update_saved_result(task_id: str, updated_result: dict):
    """Update a saved result"""
    try:
        result_file = RESULTS_DIR / f"{task_id}.json"
        if not result_file.exists():
            raise HTTPException(status_code=404, detail="Result not found")
        
        # Validate the structure
        required_fields = ["task_id", "filename", "status", "results"]
        for field in required_fields:
            if field not in updated_result:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Ensure task_id matches
        if updated_result["task_id"] != task_id:
            raise HTTPException(status_code=400, detail="Task ID mismatch")
        
        # Update the file
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(updated_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Updated saved result: {task_id}")
        return {"message": "Result updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating result {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update result")

@app.delete("/api/saved-results/{task_id}")
async def delete_saved_result(task_id: str):
    """Delete a saved result"""
    try:
        result_file = RESULTS_DIR / f"{task_id}.json"
        if result_file.exists():
            result_file.unlink()
            logger.info(f"Deleted saved result: {task_id}")
            return {"message": "Result deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Result not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting result {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete result")

# ---- Steps and Prompts Management API ----

@app.get("/api/steps")
async def get_steps_configuration():
    """Get current steps configuration"""
    try:
        steps_data = {}
        for step_id, step_config in STEPS_CONFIG.items():
            steps_data[step_id] = step_config.model_dump()
        return {
            "steps": steps_data,
            "count": len(steps_data)
        }
    except Exception as e:
        logger.error(f"Error getting steps configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to get steps configuration")

@app.post("/api/steps")
async def update_steps_configuration(config: StepsConfiguration):
    """Update the entire steps configuration"""
    global STEPS_CONFIG
    try:
        # Note: No order validation - multiple steps can have the same order (parallel processing)
        
        # Validate that all prompt files exist
        for step_id, step_config in config.steps.items():
            prompt_path = Path("prompts") / step_config.prompt_file
            if not prompt_path.exists():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Prompt file not found: {step_config.prompt_file}"
                )
        
        # Update configuration
        STEPS_CONFIG = config.steps
        save_steps_config(STEPS_CONFIG)
        
        logger.info(f"Steps configuration updated: {list(STEPS_CONFIG.keys())}")
        return {"message": "Steps configuration updated successfully", "steps": list(STEPS_CONFIG.keys())}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating steps configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to update steps configuration")

@app.post("/api/steps/{step_id}")
async def create_or_update_step(step_id: str, step_config: StepConfig):
    """Create or update a specific step (audio or LLM)"""
    global STEPS_CONFIG
    try:
        # Validate input source dependencies
        validate_input_source_dependencies(step_config, STEPS_CONFIG, step_id)
        
        # Validate based on step type
        if step_config.step_type == "llm":
            # Validate prompt file exists for LLM steps
            if not step_config.prompt_file:
                raise HTTPException(
                    status_code=400, 
                    detail="LLM steps require a prompt file"
                )
            prompt_path = Path("prompts") / step_config.prompt_file
            if not prompt_path.exists():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Prompt file not found: {step_config.prompt_file}"
                )
        elif step_config.step_type == "audio":
            # Audio steps don't need prompt files but may need parameter validation
            if not step_config.parameters:
                # Set default parameters based on step type
                step_config.parameters = get_default_audio_parameters(step_id)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown step type: {step_config.step_type}"
            )
        
        # Ensure step_id matches
        step_config.step_id = step_id
        
        # Update configuration
        STEPS_CONFIG[step_id] = step_config
        save_steps_config(STEPS_CONFIG)
        
        logger.info(f"Step '{step_id}' ({step_config.step_type}) created/updated successfully (order: {step_config.order})")
        return {"message": f"Step '{step_id}' created/updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating/updating step {step_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create/update step")

def validate_input_source_dependencies(step_config: StepConfig, current_steps: Dict[str, StepConfig], exclude_step_id: str = None):
    """Validate that the input source is available from previous steps"""
    if step_config.input_source == "transcript":
        # Check if transcript is available from an audio transcription step in a previous order
        available_steps = [
            step for step_id, step in current_steps.items() 
            if step_id != exclude_step_id and step.order < step_config.order and step.enabled
        ]
        
        has_transcript_step = any(
            step.step_type == "audio" and step.result_key == "transcript"
            for step in available_steps
        )
        
        if not has_transcript_step:
            raise HTTPException(
                status_code=400,
                detail="Input source 'transcript' requires an enabled audio transcription step in a previous order that produces the transcript result"
            )
    
    elif step_config.input_source != "file":  # "file" is valid for first audio step
        # Check if the input source result key is available from previous steps
        available_steps = [
            step for step_id, step in current_steps.items() 
            if step_id != exclude_step_id and step.order < step_config.order and step.enabled
        ]
        
        available_result_keys = [step.result_key for step in available_steps]
        
        if step_config.input_source not in available_result_keys:
            raise HTTPException(
                status_code=400,
                detail=f"Input source '{step_config.input_source}' is not available. Available sources from previous steps: {available_result_keys + (['transcript'] if any(s.step_type == 'audio' and s.result_key == 'transcript' for s in available_steps) else [])}"
            )

def get_default_audio_parameters(step_id: str) -> Dict[str, Any]:
    """Get default parameters for audio steps"""
    defaults = {
        "audio_loading": {
            "sample_rate": 16000,
            "mono": True,
            "normalize": True,
            "trim_silence": False,
            "trim_threshold_db": 20
        },
        "language_detection": {
            "detection_method": "sample",
            "sample_duration": 30,
            "sample_position": "middle",
            "default_language": "fr",
            "confidence_threshold": 0.7
        },
        "diarization": {
            "method": "vad",
            "min_segment_duration": 1.0,
            "max_segment_duration": 30.0,
            "silence_threshold": 3.0,
            "merge_threshold": 1.0,
            "vad_sensitivity": 12,
            "speaker_labels": {
                "default_first": "conseiller",
                "default_second": "client"
            },
            "max_speakers": 3,
            # Enhanced pyannote speaker context parameters
            "domain": "call_center",
            "scenario": "customer_support", 
            "expected_speakers": ["agent", "customer"],
            "speaker_roles": {
                "agent": ["conseiller", "support", "agent"],
                "customer": ["client", "customer", "caller"]
            },
            "conversation_type": "telephone",
            "language": "french",
            "context_prompt": "Call center conversation between Monaco Telecom support agent and customer",
            "min_speakers": 2,
            "speaker_change_threshold": 0.5,
            "min_speaker_duration": 2.0
        },
        "transcription": {
            "backend": "auto",
            "model_size": "large-v3",
            "use_gpu": True,
            "compute_type": "float16",
            "chunk_processing": True,
            "chunk_size": 30,
            "overlap": 5,
            "initial_prompt": "",
            "force_language": None,
            "temperature": 0.0,
            "best_of": 1,
            "beam_size": 5
        }
    }
    return defaults.get(step_id, {})

@app.delete("/api/steps/{step_id}")
async def delete_step(step_id: str):
    """Delete a step from configuration"""
    global STEPS_CONFIG
    try:
        if step_id not in STEPS_CONFIG:
            raise HTTPException(status_code=404, detail="Step not found")
        
        del STEPS_CONFIG[step_id]
        save_steps_config(STEPS_CONFIG)
        
        logger.info(f"Step '{step_id}' deleted successfully")
        return {"message": f"Step '{step_id}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting step {step_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete step")

@app.get("/api/prompts")
async def get_all_prompts():
    """Get all available prompts"""
    try:
        prompts = []
        prompts_dir = Path("prompts")
        
        if prompts_dir.exists():
            for prompt_file in prompts_dir.glob("*.txt"):
                try:
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    prompts.append({
                        "filename": prompt_file.name,
                        "content": content,
                        "size": len(content),
                        "used_by_steps": [
                            step_id for step_id, config in STEPS_CONFIG.items() 
                            if config.prompt_file == prompt_file.name
                        ]
                    })
                except Exception as e:
                    logger.error(f"Error reading prompt file {prompt_file}: {e}")
        
        return {"prompts": prompts, "count": len(prompts)}
        
    except Exception as e:
        logger.error(f"Error getting prompts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get prompts")

@app.get("/api/prompts/{filename}")
async def get_prompt(filename: str):
    """Get a specific prompt by filename"""
    try:
        prompt_path = Path("prompts") / filename
        if not prompt_path.exists():
            raise HTTPException(status_code=404, detail="Prompt file not found")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "filename": filename,
            "content": content,
            "size": len(content),
            "used_by_steps": [
                step_id for step_id, config in STEPS_CONFIG.items() 
                if config.prompt_file == filename
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get prompt")

@app.post("/api/prompts/{filename}")
async def create_or_update_prompt(filename: str, prompt_data: PromptData):
    """Create or update a prompt file"""
    try:
        # Validate filename
        if not filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="Prompt filename must end with .txt")
        
        # Ensure filename matches
        prompt_data.filename = filename
        
        # Create prompts directory if it doesn't exist
        prompts_dir = Path("prompts")
        prompts_dir.mkdir(exist_ok=True)
        
        # Write the prompt file
        prompt_path = prompts_dir / filename
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt_data.content)
        
        logger.info(f"Prompt '{filename}' created/updated successfully ({len(prompt_data.content)} characters)")
        return {"message": f"Prompt '{filename}' created/updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating/updating prompt {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create/update prompt")

@app.delete("/api/prompts/{filename}")
async def delete_prompt(filename: str):
    """Delete a prompt file"""
    try:
        prompt_path = Path("prompts") / filename
        if not prompt_path.exists():
            raise HTTPException(status_code=404, detail="Prompt file not found")
        
        # Check if prompt is used by any steps
        used_by = [step_id for step_id, config in STEPS_CONFIG.items() if config.prompt_file == filename]
        if used_by:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot delete prompt '{filename}' - it is used by steps: {', '.join(used_by)}"
            )
        
        prompt_path.unlink()
        logger.info(f"Prompt '{filename}' deleted successfully")
        return {"message": f"Prompt '{filename}' deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting prompt {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete prompt")

@app.post("/api/test-prompt")
async def test_prompt(
    prompt_content: str = None,
    prompt_filename: str = None,
    test_transcript: str = "This is a test transcript for prompt validation."
):
    """Test a prompt with Ollama without creating a full task"""
    try:
        if not prompt_content and not prompt_filename:
            raise HTTPException(status_code=400, detail="Either prompt_content or prompt_filename must be provided")
        
        if prompt_filename:
            # Load prompt from file
            prompt_content = load_prompt(prompt_filename)
        
        # Create a test task ID
        test_task_id = f"test_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Test the prompt
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                executor, 
                lambda: requests.post(
                    OLLAMA_URL, 
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": f"{prompt_content.strip()}\n\n{test_transcript.strip()}",
                        "stream": False,
                        "keep_alive": -1  # Keep model cached
                    }, 
                    timeout=60
                ).json()
            )
            
            return {
                "status": "success",
                "test_task_id": test_task_id,
                "prompt_preview": prompt_content[:200] + "..." if len(prompt_content) > 200 else prompt_content,
                "test_transcript": test_transcript,
                "ollama_response": result.get("response", "No response"),
                "model_used": OLLAMA_MODEL
            }
            
        except Exception as e:
            return {
                "status": "error",
                "test_task_id": test_task_id,
                "error": str(e),
                "prompt_preview": prompt_content[:200] + "..." if len(prompt_content) > 200 else prompt_content
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing prompt: {e}")
        raise HTTPException(status_code=500, detail="Failed to test prompt")

@app.get("/api/audio-step-templates")
async def get_audio_step_templates():
    """Get available audio step templates for creating new steps"""
    templates = {
        "audio_loading": {
            "name": "Audio Loading & Preprocessing",
            "description": "Load and preprocess audio file",
            "parameters": get_default_audio_parameters("audio_loading"),
            "result_key": "audio_metadata",
            "input_source": "file"
        },
        "language_detection": {
            "name": "Language Detection",
            "description": "Detect dominant language in audio",
            "parameters": get_default_audio_parameters("language_detection"),
            "result_key": "language_info",
            "input_source": "audio_metadata"
        },
        "diarization": {
            "name": "Speaker Diarization",
            "description": "Identify and segment speakers",
            "parameters": get_default_audio_parameters("diarization"),
            "result_key": "diarization_segments",
            "input_source": "audio_metadata"
        },
        "transcription": {
            "name": "Audio Transcription",
            "description": "Convert speech to text",
            "parameters": get_default_audio_parameters("transcription"),
            "result_key": "transcript",
            "input_source": "diarization_segments"
        },
        "audio_transcription": {
            "name": "Audio Transcription",
            "description": "Convert speech to text",
            "parameters": get_default_audio_parameters("transcription"),
            "result_key": "transcript",
            "input_source": "diarization_segments"
        }
    }
    return {"templates": templates}

@app.get("/api/parameter-schemas")
async def get_parameter_schemas():
    """Get parameter schemas for different step types"""
    schemas = {
        "audio_loading": {
            "sample_rate": {"type": "number", "default": 16000, "description": "Target sample rate in Hz"},
            "mono": {"type": "boolean", "default": True, "description": "Convert to mono audio"},
            "normalize": {"type": "boolean", "default": True, "description": "Normalize audio volume"},
            "trim_silence": {"type": "boolean", "default": False, "description": "Trim silence from start/end"},
            "trim_threshold_db": {"type": "number", "default": 20, "description": "Silence threshold in dB"}
        },
        "language_detection": {
            "detection_method": {"type": "select", "options": ["sample", "full"], "default": "sample", "description": "Detection method"},
            "sample_duration": {"type": "number", "default": 30, "description": "Sample duration in seconds"},
            "sample_position": {"type": "select", "options": ["start", "middle", "end"], "default": "middle", "description": "Sample position"},
            "default_language": {"type": "text", "default": "fr", "description": "Default language code"},
            "confidence_threshold": {"type": "number", "min": 0, "max": 1, "step": 0.1, "default": 0.7, "description": "Confidence threshold"}
        },
        "diarization": {
            "method": {"type": "select", "options": ["vad", "pyannote", "disabled"], "default": "vad", "description": "Diarization method"},
            "min_segment_duration": {"type": "number", "default": 1.0, "description": "Minimum segment duration in seconds"},
            "max_segment_duration": {"type": "number", "default": 30.0, "description": "Maximum segment duration in seconds"},
            "silence_threshold": {"type": "number", "default": 3.0, "description": "Silence threshold for speaker change in seconds"},
            "merge_threshold": {"type": "number", "default": 1.0, "description": "Threshold for merging close segments in seconds"},
            "vad_sensitivity": {"type": "number", "default": 12, "description": "VAD sensitivity (top_db)"},
            "max_speakers": {"type": "number", "default": 3, "description": "Maximum number of speakers"},
            "min_speakers": {"type": "number", "default": 2, "description": "Minimum number of speakers (pyannote)"},
            "min_speaker_duration": {"type": "number", "default": 2.0, "description": "Minimum speaker duration in seconds (pyannote)"},
            "speaker_change_threshold": {"type": "number", "min": 0, "max": 1, "step": 0.1, "default": 0.5, "description": "Speaker change confidence threshold (pyannote)"},
            "domain": {"type": "select", "options": ["call_center", "meeting", "interview", "general"], "default": "call_center", "description": "Conversation domain context"},
            "scenario": {"type": "select", "options": ["customer_support", "sales", "technical_support", "general"], "default": "customer_support", "description": "Conversation scenario"},
            "conversation_type": {"type": "select", "options": ["telephone", "video_call", "in_person", "webinar"], "default": "telephone", "description": "Type of conversation"},
            "language": {"type": "select", "options": ["french", "english", "spanish", "italian"], "default": "french", "description": "Primary conversation language"},
            "context_prompt": {"type": "textarea", "default": "Call center conversation between Monaco Telecom support agent and customer", "description": "Context description for better speaker identification"},
            "expected_speakers": {"type": "text", "default": "agent,customer", "description": "Expected speaker roles (comma-separated)"}
        },
        "transcription": {
            "backend": {"type": "select", "options": ["auto", "faster-whisper", "openai-whisper"], "default": "auto", "description": "Transcription backend"},
            "model_size": {"type": "select", "options": ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"], "default": "large-v3", "description": "Model size"},
            "use_gpu": {"type": "boolean", "default": True, "description": "Use GPU acceleration"},
            "compute_type": {"type": "select", "options": ["float16", "float32", "int8"], "default": "float16", "description": "Compute precision"},
            "chunk_processing": {"type": "boolean", "default": True, "description": "Process in chunks"},
            "chunk_size": {"type": "number", "default": 30, "description": "Chunk size in seconds"},
            "overlap": {"type": "number", "default": 5, "description": "Chunk overlap in seconds"},
            "initial_prompt": {"type": "textarea", "default": "", "description": "Context prompt to guide transcription (e.g., 'The following is a call center conversation about insurance claims.')"},
            "force_language": {"type": "text", "default": "", "description": "Force specific language (leave empty for auto-detection)"},
            "temperature": {"type": "number", "min": 0, "max": 1, "step": 0.1, "default": 0.0, "description": "Temperature for sampling"},
            "best_of": {"type": "number", "default": 1, "description": "Number of candidates to generate"},
            "beam_size": {"type": "number", "default": 5, "description": "Beam size for decoding"}
        }
    }
    return {"schemas": schemas}

@app.get("/api/available-input-sources")
async def get_available_input_sources(order: int = None):
    """Get available input sources for a given step order"""
    try:
        if order is None:
            return {"input_sources": []}
        
        # Get steps with lower order numbers that are enabled
        available_steps = [
            step for step in STEPS_CONFIG.values() 
            if step.order < order and step.enabled
        ]
        
        input_sources = []
        
        # Check if transcript is available from audio transcription step
        has_transcript_step = any(
            step.step_type == "audio" and step.result_key == "transcript"
            for step in available_steps
        )
        
        if has_transcript_step:
            input_sources.append({
                "value": "transcript",
                "label": "Full Transcript (from audio transcription)",
                "type": "transcript"
            })
        
        # Add other result keys from previous steps
        for step in available_steps:
            if step.result_key and step.result_key != "transcript":
                input_sources.append({
                    "value": step.result_key,
                    "label": f"{step.name} Result ({step.result_key}) - Order {step.order}",
                    "type": "step_result",
                    "source_step": step.step_id,
                    "source_order": step.order
                })
        
        # Add "file" option for first audio steps (order 1)
        if order == 1:
            input_sources.insert(0, {
                "value": "file",
                "label": "Audio File (direct input)",
                "type": "file"
            })
        
        return {"input_sources": input_sources}
        
    except Exception as e:
        logger.error(f"Error getting available input sources: {e}")
        raise HTTPException(status_code=500, detail="Failed to get available input sources")

def get_step_input_text(step_config: StepConfig, task_state: TaskState, transcript: str, task_id: str) -> str:
    """Determine the input text for a step based on its configuration"""
    if step_config.input_source == "transcript":
        logger.info(f"[{task_id}] Step {step_config.step_id} using full transcript ({len(transcript)} characters)")
        return transcript
    
    # Try to get the result from the specified input source
    source_result = task_state.results.get(step_config.input_source)
    
    if source_result:
        # Handle different result formats
        if isinstance(source_result, dict):
            # Extract text from result object - try common keys
            result_text = (
                source_result.get("response", "") or 
                source_result.get("summary", "") or 
                source_result.get("text", "") or 
                str(source_result)
            )
        elif isinstance(source_result, str):
            result_text = source_result
        else:
            result_text = str(source_result)
        
        if result_text and len(result_text.strip()) > 50:  # Minimum reasonable length
            logger.info(f"[{task_id}] Step {step_config.step_id} using {step_config.input_source} result ({len(result_text)} characters)")
            return result_text
    
    # Fallback to transcript if the source is not available or too short
    logger.warning(f"[{task_id}] Step {step_config.step_id} falling back to full transcript - {step_config.input_source} not available or too short")
    return transcript

def process_audio_task(temp_file_path: str, task_id: str, filename: str):
    """Process audio file with modular step-based pipeline"""
    try:
        # Get the task state
        task_state = get_task_state(task_id)
        
        # Step 1: Process audio steps (loading, language detection, diarization, transcription)
        audio_steps = {k: v for k, v in STEPS_CONFIG.items() if v.step_type == "audio" and v.enabled}
        if audio_steps:
            logger.info(f"[{task_id}] Starting audio processing pipeline with {len(audio_steps)} steps")
            
            # Create AudioProcessor with current steps configuration
            from whisper_wrapper import AudioProcessor
            steps_config_dict = {k: v.model_dump() for k, v in STEPS_CONFIG.items()}
            audio_processor = AudioProcessor(steps_config_dict)
            
            # Process through audio steps
            audio_results = audio_processor.process_audio_steps(temp_file_path, task_id, update_task_step)
            
            # Extract transcript from results - handle different result key formats
            transcript = None
            transcript_data = None
            
            # Look for transcript in audio_results using the configured result key
            for step_id, step_config in audio_steps.items():
                if step_config.result_key == "transcript":
                    transcript_data = audio_results.get(step_config.result_key)
                    break
            
            # If not found, try the generic "transcript" key
            if not transcript_data:
                transcript_data = audio_results.get("transcript")
            
            if transcript_data:
                if isinstance(transcript_data, dict):
                    transcript = transcript_data.get("text", transcript_data.get("transcript", ""))
                elif isinstance(transcript_data, str):
                    transcript = transcript_data
                else:
                    transcript = str(transcript_data)
                
                # Store additional audio processing results
                task_state.results.update({
                    "audio_metadata": audio_results.get("audio_metadata"),
                    "language_info": audio_results.get("language_info"), 
                    "diarization_segments": audio_results.get("diarization_segments"),
                    "transcript": transcript,  # Ensure transcript is stored with standard key
                    "transcript_metadata": transcript_data.get("metadata", {}) if isinstance(transcript_data, dict) else {}
                })
                
                logger.info(f"[{task_id}] Audio processing completed - transcript length: {len(transcript)} characters")
            else:
                logger.error(f"[{task_id}] No transcript produced from audio processing")
                logger.error(f"[{task_id}] Available audio_results keys: {list(audio_results.keys())}")
                return
        else:
            # Fallback to legacy transcription
            logger.warning(f"[{task_id}] No audio steps configured, using legacy transcription")
            transcript = transcribe_audio(temp_file_path, task_id)
        
        # Step 2: Run enabled LLM analyses in order groups (exclude audio steps)
        llm_steps = {k: v for k, v in STEPS_CONFIG.items() if v.enabled and v.step_type == "llm"}
        if not llm_steps:
            logger.info(f"[{task_id}] No LLM analysis steps enabled, completing task")
            task_state.status = "completed"
            task_state.updated_at = datetime.now().isoformat()
            save_completed_task(task_state)
            return
        
        # Group LLM steps by order
        order_groups = {}
        for step_id, step_config in llm_steps.items():
            order = step_config.order
            if order not in order_groups:
                order_groups[order] = []
            order_groups[order].append((step_id, step_config))
        
        # Sort orders to process sequentially
        sorted_orders = sorted(order_groups.keys())
        logger.info(f"[{task_id}] Processing {len(llm_steps)} LLM steps in {len(sorted_orders)} order groups: {sorted_orders}")
        
        # Process each order group sequentially
        for order in sorted_orders:
            steps_in_order = order_groups[order]
            logger.info(f"[{task_id}] Processing order {order} with {len(steps_in_order)} LLM steps: {[s[0] for s in steps_in_order]}")
            
            # Run all steps in this order group in parallel
            with ThreadPoolExecutor(max_workers=len(steps_in_order)) as executor:
                futures = {}
                
                for step_id, step_config in steps_in_order:
                    try:
                        # Skip audio steps (they were already processed)
                        if step_config.step_type == "audio":
                            logger.info(f"[{task_id}] Skipping audio step {step_id} - already processed")
                            continue
                            
                        # Determine input text for this specific step based on its configuration
                        input_text = get_step_input_text(step_config, task_state, transcript, task_id)
                        
                        # Load prompt dynamically
                        prompt = load_prompt(step_config.prompt_file)
                        future = executor.submit(query_ollama, prompt, input_text, task_id, step_id)
                        futures[step_id] = future
                        
                        source_description = step_config.input_source if step_config.input_source != "transcript" else "transcript"
                        logger.info(f"[{task_id}] Submitted {step_id} (order {order}) for processing with {source_description} ({len(input_text)} characters)")
                    except Exception as e:
                        logger.error(f"[{task_id}] Failed to submit {step_id}: {e}")
                        update_task_step(task_id, step_id, "error", 0, f"Failed to start: {str(e)}")
                
                # Wait for all steps in this order group to complete
                for step_id, future in futures.items():
                    try:
                        result = future.result(timeout=400)  # Slightly longer than Ollama timeout
                        logger.info(f"[{task_id}] {step_id} (order {order}) completed successfully")
                    except Exception as e:
                        logger.error(f"[{task_id}] {step_id} (order {order}) failed: {e}")
                        update_task_step(task_id, step_id, "error", 0, f"Processing failed: {str(e)}")
            
            logger.info(f"[{task_id}] Order {order} completed, moving to next order group")
        
        # Final status update
        task_state.status = "completed"
        task_state.updated_at = datetime.now().isoformat()
        logger.info(f"[{task_id}] All processing completed")
        
        # Save completed task to file
        save_completed_task(task_state)
        
    except Exception as e:
        # Mark task as failed
        task_state = get_task_state(task_id)
        task_state.status = "failed"
        task_state.error_message = str(e)
        task_state.updated_at = datetime.now().isoformat()
        logger.error(f"[{task_id}] Processing failed: {e}")
        
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"[{task_id}] Temporary file cleaned up")
            except Exception as e:
                logger.error(f"[{task_id}] Failed to clean up temporary file: {e}")

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    """Upload an audio file for full analysis"""
    
    # Generate task ID
    task_id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    
    # Validate file type
    allowed_extensions = {'.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg'}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Create task state
        task_state = create_task_state(task_id, file.filename)
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Start background processing
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, process_audio_task, temp_file_path, task_id, file.filename)
        
        # Return task ID immediately
        return JSONResponse(content={
            "task_id": task_id,
            "filename": file.filename,
            "status": "processing",
            "message": "Processing started, check progress endpoint for updates"
        })
        
    except Exception as e:
        error_msg = f"Failed to start processing: {str(e)}"
        logger.error(f"[{task_id}] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/transcribe-only")
async def transcribe_only(file: UploadFile = File(...)):
    """Upload an audio file for transcription only"""
    
    # Generate task ID
    task_id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    
    # Validate file type
    allowed_extensions = {'.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg'}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Create task state (only transcription step)
        task_state = create_task_state(task_id, file.filename)
        # Remove non-transcription steps for transcribe-only mode
        task_state.steps = {
            "transcription": {"status": "pending", "progress": 0, "message": "Waiting to start...", "details": {}}
        }
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Transcribe directly (synchronous for transcribe-only)
        try:
            transcript = transcribe_audio(temp_file_path, task_id)
            task_state.status = "completed"
            task_state.updated_at = datetime.now().isoformat()
            
            # Save completed task to file
            save_completed_task(task_state)
            
            # Clean up
            os.unlink(temp_file_path)
            
            return JSONResponse(content={
                "task_id": task_id,
                "filename": file.filename,
                "transcript": transcript,
                "status": "completed"
            })
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise
        
    except Exception as e:
        error_msg = f"Transcription failed: {str(e)}"
        logger.error(f"[{task_id}] {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

def save_completed_task(task_state: TaskState):
    """Save completed task to file for persistence"""
    try:
        if task_state.status == "completed":
            result_data = {
                "task_id": task_state.task_id,
                "filename": task_state.filename,
                "status": task_state.status,
                "completed_at": task_state.updated_at,
                "results": task_state.results,
                "processing_steps": task_state.steps
            }
            
            result_file = RESULTS_DIR / f"{task_state.task_id}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[{task_state.task_id}] Results saved to {result_file}")
            
    except Exception as e:
        logger.error(f"Failed to save task {task_state.task_id}: {e}")

def load_saved_results():
    """Load all saved results from files"""
    saved_results = []
    try:
        for result_file in RESULTS_DIR.glob("*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    saved_results.append(result_data)
            except Exception as e:
                logger.error(f"Failed to load result file {result_file}: {e}")
        
        # Sort by completion time (newest first)
        saved_results.sort(key=lambda x: x.get('completed_at', ''), reverse=True)
        logger.info(f"Loaded {len(saved_results)} saved results")
        
    except Exception as e:
        logger.error(f"Failed to load saved results: {e}")
    
    return saved_results

def get_saved_result(task_id: str):
    """Get a specific saved result by task ID"""
    try:
        result_file = RESULTS_DIR / f"{task_id}.json"
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load result {task_id}: {e}")
    return None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 