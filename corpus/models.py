# models.py - Fixed Version
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Generator, List
from dataclasses import dataclass
from enum import Enum
import requests
import json
import socket
import time
import subprocess
import threading
from urllib.parse import urlparse
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class ModelType(Enum):
    """Supported model types"""
    OPENAI = "openai"
    GEMINI = "gemini"
    PHI_OLLAMA = "phi_ollama"
    PHI_VLLM = "phi_vllm"
    VLLM_REMOTE = "vllm_remote"
    OPENAI_COMPATIBLE = "openai_compatible"

class ComputeLocation(Enum):
    """Where the compute happens"""
    LOCAL = "local"
    LAN = "lan"
    CLOUD = "cloud"

@dataclass
class ModelConfig:
    """Configuration for a model"""
    name: str
    model_type: ModelType
    model_id: str
    compute_location: ComputeLocation
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    host: str = "localhost"
    port: int = 8000
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.8
    context_length: int = 4096
    supports_streaming: bool = True
    supports_system_prompt: bool = True
    timeout: int = 30
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    quantization: Optional[str] = None

class BaseModel(ABC):
    """Abstract base class for all model implementations"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.name
        self.model_id = config.model_id
        self._is_initialized = False
        
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate a response from the model"""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Generator[str, None, None]:
        """Generate a streaming response from the model"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available and working"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "type": self.config.model_type.value,
            "location": self.config.compute_location.value,
            "context_length": self.config.context_length,
            "supports_streaming": self.config.supports_streaming,
            "endpoint": f"{self.config.host}:{self.config.port}" if self.config.compute_location != ComputeLocation.CLOUD else "cloud"
        }

class OpenAIModel(BaseModel):
    """OpenAI model implementation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Don't require API key during initialization for debugging
        self.api_key = config.api_key
        self.api_url = config.base_url or "https://api.openai.com/v1"
        
        # Initialize headers if API key is available
        if self.api_key:
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        else:
            self.headers = {"Content-Type": "application/json"}
    
    def _check_api_key(self):
        """Check if API key is available"""
        if not self.api_key:
            raise ValueError(f"API key required for OpenAI model {self.config.name}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response using OpenAI API"""
        self._check_api_key()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p)
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response using OpenAI API"""
        self._check_api_key()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                        except:
                            continue
        except Exception as e:
            raise Exception(f"OpenAI streaming error: {e}")
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available"""
        if not self.api_key:
            return False
            
        try:
            response = requests.get(
                f"{self.api_url}/models",
                headers=self.headers,
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

class GeminiModel(BaseModel):
    """Google Gemini model implementation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.api_key = config.api_key
        self.model = None
        
        # Only initialize if API key is available
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(config.model_id)
            except ImportError:
                console.print("[yellow]google-generativeai not installed. Run: pip install google-generativeai[/yellow]")
                raise ValueError("google-generativeai package required for Gemini models")
            except Exception as e:
                console.print(f"[yellow]Failed to initialize Gemini model: {e}[/yellow]")
    
    def _check_api_key(self):
        """Check if API key is available"""
        if not self.api_key or not self.model:
            raise ValueError(f"API key and model initialization required for Gemini model {self.config.name}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response using Gemini API"""
        self._check_api_key()
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        
        try:
            import google.generativeai as genai
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", self.config.temperature),
                    top_p=kwargs.get("top_p", self.config.top_p),
                    max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                )
            )
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response using Gemini API"""
        self._check_api_key()
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        
        try:
            import google.generativeai as genai
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", self.config.temperature),
                    top_p=kwargs.get("top_p", self.config.top_p),
                    max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                ),
                stream=True
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise Exception(f"Gemini streaming error: {e}")
    
    def is_available(self) -> bool:
        """Check if Gemini API is available"""
        if not self.api_key or not self.model:
            return False
            
        try:
            test_response = self.model.generate_content("Hi")
            return bool(test_response.text)
        except:
            return False

class PhiOllamaModel(BaseModel):
    """Phi-3 model via Ollama"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_url = f"http://{config.host}:{config.port or 11434}"
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response using Ollama API"""
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens)
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            raise Exception(f"Ollama API error: {e}")
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response using Ollama API"""
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p)
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    if "response" in chunk:
                        yield chunk["response"]
                    if chunk.get("done", False):
                        break
        except Exception as e:
            raise Exception(f"Ollama streaming error: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama is available and model is installed"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Check if our model is available
            models = response.json().get("models", [])
            return any(model["name"].startswith(self.model_id) for model in models)
        except:
            return False
    
    def pull_model(self) -> bool:
        """Pull/download the Phi-3 model in Ollama"""
        try:
            console.print(f"[cyan]Pulling {self.model_id} model...[/cyan]")
            
            payload = {"name": self.model_id}
            response = requests.post(
                f"{self.api_url}/api/pull",
                json=payload,
                stream=True,
                timeout=300
            )
            
            for line in response.iter_lines():
                if line:
                    status = json.loads(line.decode('utf-8'))
                    if "status" in status:
                        console.print(f"[dim]{status['status']}[/dim]")
                    if status.get("status") == "success":
                        console.print(f"[green]✓ Successfully pulled {self.model_id}[/green]")
                        return True
            
            return False
        except Exception as e:
            console.print(f"[red]Failed to pull model: {e}[/red]")
            return False

class VLLMManager:
    """Manager for vLLM server instances"""
    
    @staticmethod
    def install_vllm() -> bool:
        """Install vLLM via pip"""
        try:
            console.print("[cyan]Installing vLLM...[/cyan]")
            result = subprocess.run(
                ["pip", "install", "vllm", "transformers", "torch"],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                console.print("[green]✓ vLLM installed successfully[/green]")
                return True
            else:
                console.print(f"[red]vLLM installation failed: {result.stderr}[/red]")
                return False
        except Exception as e:
            console.print(f"[red]Error installing vLLM: {e}[/red]")
            return False
    
    @staticmethod
    def start_vllm_server(config: ModelConfig) -> Optional[subprocess.Popen]:
        """Start a vLLM server instance"""
        try:
            console.print(f"[cyan]Starting vLLM server for {config.model_id}...[/cyan]")
            
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", config.model_id,
                "--host", config.host,
                "--port", str(config.port),
                "--gpu-memory-utilization", str(config.gpu_memory_utilization),
                "--tensor-parallel-size", str(config.tensor_parallel_size)
            ]
            
            if config.quantization:
                cmd.extend(["--quantization", config.quantization])
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a bit for server to start
            time.sleep(10)
            
            # Check if server is responding
            try:
                response = requests.get(f"http://{config.host}:{config.port}/v1/models", timeout=5)
                if response.status_code == 200:
                    console.print(f"[green]✓ vLLM server started successfully on {config.host}:{config.port}[/green]")
                    return process
            except:
                pass
            
            console.print("[red]Failed to start vLLM server[/red]")
            process.terminate()
            return None
            
        except Exception as e:
            console.print(f"[red]Error starting vLLM server: {e}[/red]")
            return None

class PhiVLLMModel(BaseModel):
    """Phi-3 model via vLLM (OpenAI-compatible API)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_url = f"http://{config.host}:{config.port}/v1"
        self.headers = {"Content-Type": "application/json"}
        self.vllm_process = None
    
    def start_server(self) -> bool:
        """Start local vLLM server if needed"""
        if self.config.compute_location == ComputeLocation.LOCAL:
            self.vllm_process = VLLMManager.start_vllm_server(self.config)
            return self.vllm_process is not None
        return True
    
    def stop_server(self):
        """Stop local vLLM server"""
        if self.vllm_process:
            self.vllm_process.terminate()
            self.vllm_process = None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response using vLLM OpenAI-compatible API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p)
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=120           )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"vLLM API error: {e}")
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response using vLLM API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                        except:
                            continue
        except Exception as e:
            raise Exception(f"vLLM streaming error: {e}")
    
    def is_available(self) -> bool:
        """Check if vLLM server is available"""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=5)
            return response.status_code == 200
        except:
            return False