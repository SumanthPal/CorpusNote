# models_manager.py - Integrated with Config Manager
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from .models import (
    BaseModel, ModelConfig, ModelType, ComputeLocation,
    OpenAIModel, GeminiModel, PhiOllamaModel, PhiVLLMModel, VLLMManager
)

console = Console()

class PlaceholderModel(BaseModel):
    """Placeholder model for unconfigured models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.name
        self.model_id = config.model_id
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        raise Exception(f"Model {self.name} is not configured. Please set up API key first.")
    
    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        raise Exception(f"Model {self.name} is not configured. Please set up API key first.")
    
    def is_available(self) -> bool:
        return False

class ModelsManager:
    """Manages all available models and their configurations"""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path.home() / ".corpus" / "models.json"
        self.models: Dict[str, BaseModel] = {}
        self.active_model: Optional[str] = None
        self._ensure_config_dir()
        self.load_configurations()
    
    def _ensure_config_dir(self):
        """Ensure config directory exists"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _get_api_key(self, model_type: str) -> Optional[str]:
        """Get API key from config manager"""
        try:
            from corpus.config_manager import get_config
            config = get_config()
            
            if model_type == "openai":
                return config.get("OPENAI_API_KEY")
            elif model_type == "gemini":
                return config.get("GEMINI_API_KEY")
            return None
        except ImportError:
            console.print("[yellow]Config manager not available, using stored keys[/yellow]")
            return None
    
    def load_configurations(self):
        """Load model configurations from file"""
        console.print(f"[blue]Loading config from: {self.config_file}[/blue]")
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    console.print(f"[blue]Config data loaded: {list(data.get('models', {}).keys())}[/blue]")
                    self._load_models_from_config(data)
                    self.active_model = data.get("active_model")
                    console.print(f"[blue]Models after loading: {list(self.models.keys())}[/blue]")
            except Exception as e:
                console.print(f"[red]Error loading model config: {e}[/red]")
                self._create_default_config()
        else:
            console.print("[yellow]Config file doesn't exist, creating default...[/yellow]")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default model configurations"""
        console.print("[yellow]Creating default model configuration...[/yellow]")
        
        default_configs = {
            "models": {
                "gpt-4": {
                    "name": "gpt-4",
                    "model_type": "openai",
                    "model_id": "gpt-4",
                    "compute_location": "cloud",
                    "max_tokens": 4096,
                    "context_length": 8192
                },
                "gpt-3.5-turbo": {
                    "name": "gpt-3.5-turbo",
                    "model_type": "openai",
                    "model_id": "gpt-3.5-turbo",
                    "compute_location": "cloud",
                    "max_tokens": 4096,
                    "context_length": 4096
                },
                "gemini-pro": {
                    "name": "gemini-pro",
                    "model_type": "gemini",
                    "model_id": "gemini-1.5-flash",
                    "compute_location": "cloud",
                    "max_tokens": 2048,
                    "context_length": 32768
                },
                "phi3-mini-ollama": {
                    "name": "phi3-mini-ollama",
                    "model_type": "phi_ollama",
                    "model_id": "phi3:mini",
                    "compute_location": "local",
                    "host": "localhost",
                    "port": 11434,
                    "max_tokens": 2048,
                    "context_length": 4096
                },
                "phi3-mini-vllm": {
                    "name": "phi3-mini-vllm",
                    "model_type": "phi_vllm",
                    "model_id": "microsoft/Phi-3-mini-4k-instruct",
                    "compute_location": "local",
                    "host": "localhost",
                    "port": 8000,
                    "max_tokens": 2048,
                    "context_length": 4096,
                    "gpu_memory_utilization": 0.8,
                    "tensor_parallel_size": 1
                }
            },
            "active_model": "gemini-pro"
        }
        
        self.save_configurations(default_configs)
        # After saving, load the models
        self._load_models_from_config(default_configs)
    
    def _load_models_from_config(self, config_data: Dict):
        """Load models from configuration data"""
        console.print(f"[blue]Loading models from config data...[/blue]")
        
        for model_name, model_data in config_data.get("models", {}).items():
            console.print(f"[blue]Processing model: {model_name} (type: {model_data.get('model_type')})[/blue]")
            
            try:
                # Get API key from config manager for cloud models
                api_key = None
                if model_data.get("model_type") in ["openai", "gemini"]:
                    api_key = self._get_api_key(model_data["model_type"])
                    console.print(f"[blue]API key for {model_name}: {'Found' if api_key else 'Not found'}[/blue]")
                
                model_config = ModelConfig(
                    name=model_data["name"],
                    model_type=ModelType(model_data["model_type"]),
                    model_id=model_data["model_id"],
                    compute_location=ComputeLocation(model_data["compute_location"]),
                    api_key=api_key,  # Use API key from config manager
                    base_url=model_data.get("base_url"),
                    host=model_data.get("host", "localhost"),
                    port=model_data.get("port", 8000),
                    max_tokens=model_data.get("max_tokens", 2048),
                    temperature=model_data.get("temperature", 0.7),
                    top_p=model_data.get("top_p", 0.8),
                    context_length=model_data.get("context_length", 4096),
                    gpu_memory_utilization=model_data.get("gpu_memory_utilization", 0.8),
                    tensor_parallel_size=model_data.get("tensor_parallel_size", 1),
                    quantization=model_data.get("quantization")
                )
                
                console.print(f"[blue]Created config for {model_name}, API key present: {bool(model_config.api_key)}[/blue]")
                
                # Create model instance based on type
                if model_config.model_type == ModelType.OPENAI:
                    if model_config.api_key:
                        console.print(f"[green]Creating OpenAI model: {model_name}[/green]")
                        self.models[model_name] = OpenAIModel(model_config)
                    else:
                        console.print(f"[yellow]Creating placeholder for OpenAI model: {model_name}[/yellow]")
                        self.models[model_name] = PlaceholderModel(model_config)
                        
                elif model_config.model_type == ModelType.GEMINI:
                    if model_config.api_key:
                        console.print(f"[green]Creating Gemini model: {model_name}[/green]")
                        self.models[model_name] = GeminiModel(model_config)
                    else:
                        console.print(f"[yellow]Creating placeholder for Gemini model: {model_name}[/yellow]")
                        self.models[model_name] = PlaceholderModel(model_config)
                        
                elif model_config.model_type == ModelType.PHI_OLLAMA:
                    console.print(f"[green]Creating Phi Ollama model: {model_name}[/green]")
                    self.models[model_name] = PhiOllamaModel(model_config)
                    
                elif model_config.model_type == ModelType.PHI_VLLM:
                    console.print(f"[green]Creating Phi vLLM model: {model_name}[/green]")
                    self.models[model_name] = PhiVLLMModel(model_config)
                    
                console.print(f"[green]✓ Successfully loaded model: {model_name}[/green]")
                    
            except Exception as e:
                console.print(f"[red]Error loading model {model_name}: {e}[/red]")
                import traceback
                console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        
        console.print(f"[blue]Final models loaded: {list(self.models.keys())}[/blue]")
    
    def refresh_models(self):
        """Refresh all models to pick up new API keys from config"""
        console.print("[cyan]Refreshing models with current API keys...[/cyan]")
        
        # Get current API keys
        openai_key = self._get_api_key("openai")
        gemini_key = self._get_api_key("gemini")
        
        updated_count = 0
        
        for model_name, model in list(self.models.items()):
            model_type = model.config.model_type
            
            if model_type == ModelType.OPENAI:
                if openai_key and isinstance(model, PlaceholderModel):
                    # Convert placeholder to real model
                    model.config.api_key = openai_key
                    try:
                        self.models[model_name] = OpenAIModel(model.config)
                        console.print(f"[green]✓ Activated OpenAI model: {model_name}[/green]")
                        updated_count += 1
                    except Exception as e:
                        console.print(f"[red]Failed to activate {model_name}: {e}[/red]")
                elif openai_key and hasattr(model, 'headers'):
                    # Update existing model
                    model.config.api_key = openai_key
                    model.headers = {
                        "Authorization": f"Bearer {openai_key}",
                        "Content-Type": "application/json"
                    }
                    console.print(f"[green]✓ Updated OpenAI model: {model_name}[/green]")
                    updated_count += 1
                    
            elif model_type == ModelType.GEMINI:
                if gemini_key and isinstance(model, PlaceholderModel):
                    # Convert placeholder to real model
                    model.config.api_key = gemini_key
                    try:
                        self.models[model_name] = GeminiModel(model.config)
                        console.print(f"[green]✓ Activated Gemini model: {model_name}[/green]")
                        updated_count += 1
                    except Exception as e:
                        console.print(f"[red]Failed to activate {model_name}: {e}[/red]")
                elif gemini_key and hasattr(model, 'model'):
                    # Update existing model
                    model.config.api_key = gemini_key
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=gemini_key)
                        model.model = genai.GenerativeModel(model.config.model_id)
                        console.print(f"[green]✓ Updated Gemini model: {model_name}[/green]")
                        updated_count += 1
                    except Exception as e:
                        console.print(f"[red]Failed to update {model_name}: {e}[/red]")
        
        if updated_count > 0:
            console.print(f"[green]✓ Refreshed {updated_count} models[/green]")
        else:
            console.print("[yellow]No models needed updating[/yellow]")
    
    def save_configurations(self, config_data: Optional[Dict] = None):
        """Save model configurations to file (without API keys)"""
        console.print(f"[blue]Saving configurations to: {self.config_file}[/blue]")
        
        if config_data is None:
            # Build config from current models (excluding API keys)
            config_data = {
                "models": {},
                "active_model": self.active_model
            }
            
            for name, model in self.models.items():
                config_data["models"][name] = {
                    "name": model.config.name,
                    "model_type": model.config.model_type.value,
                    "model_id": model.config.model_id,
                    "compute_location": model.config.compute_location.value,
                    "base_url": model.config.base_url,
                    "host": model.config.host,
                    "port": model.config.port,
                    "max_tokens": model.config.max_tokens,
                    "temperature": model.config.temperature,
                    "top_p": model.config.top_p,
                    "context_length": model.config.context_length,
                    "gpu_memory_utilization": model.config.gpu_memory_utilization,
                    "tensor_parallel_size": model.config.tensor_parallel_size,
                    "quantization": model.config.quantization
                    # Note: API keys are NOT saved here - they come from config manager
                }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            console.print(f"[green]✓ Configuration saved successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error saving model config: {e}[/red]")
    
    def add_model(self, config: ModelConfig) -> bool:
        """Add a new model"""
        try:
            if config.model_type == ModelType.OPENAI:
                model = OpenAIModel(config)
            elif config.model_type == ModelType.GEMINI:
                model = GeminiModel(config)
            elif config.model_type == ModelType.PHI_OLLAMA:
                model = PhiOllamaModel(config)
            elif config.model_type == ModelType.PHI_VLLM:
                model = PhiVLLMModel(config)
            else:
                console.print(f"[red]Unsupported model type: {config.model_type}[/red]")
                return False
            
            self.models[config.name] = model
            self.save_configurations()
            console.print(f"[green]✓ Added model: {config.name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error adding model: {e}[/red]")
            return False
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model"""
        if model_name in self.models:
            # Stop server if it's a local vLLM model
            model = self.models[model_name]
            if isinstance(model, PhiVLLMModel):
                model.stop_server()
            
            del self.models[model_name]
            
            # Update active model if removed
            if self.active_model == model_name:
                self.active_model = next(iter(self.models.keys())) if self.models else None
            
            self.save_configurations()
            console.print(f"[green]✓ Removed model: {model_name}[/green]")
            return True
        else:
            console.print(f"[red]Model not found: {model_name}[/red]")
            return False
    
    def set_active_model(self, model_name: str) -> bool:
        """Set the active model"""
        if model_name in self.models:
            self.active_model = model_name
            self.save_configurations()
            console.print(f"[green]✓ Active model set to: {model_name}[/green]")
            return True
        else:
            console.print(f"[red]Model not found: {model_name}[/red]")
            return False
    
    def get_active_model(self) -> Optional[BaseModel]:
        """Get the currently active model"""
        if self.active_model and self.active_model in self.models:
            model = self.models[self.active_model]
            # Check if it's a placeholder model
            if isinstance(model, PlaceholderModel):
                return None
            return model
        return None
    
    def list_models(self) -> Table:
        """List all available models in a table"""
        console.print(f"[blue]Listing {len(self.models)} models[/blue]")
        
        table = Table(title="Available Models", show_lines=True)
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Model ID", style="green")
        table.add_column("Location", style="yellow")
        table.add_column("Status", style="blue")
        table.add_column("Active", style="red")
        
        for name, model in self.models.items():
            console.print(f"[blue]Processing model for table: {name} (type: {type(model).__name__})[/blue]")
            
            if isinstance(model, PlaceholderModel):
                status = "🔧 Needs Setup"
                available = False
            else:
                try:
                    available = model.is_available()
                    status = "🟢 Available" if available else "🔴 Unavailable"
                except Exception as e:
                    console.print(f"[yellow]Error checking availability for {name}: {e}[/yellow]")
                    status = "❓ Unknown"
            
            active = "✓" if name == self.active_model else ""
            
            table.add_row(
                name,
                model.config.model_type.value,
                model.model_id if hasattr(model, 'model_id') else model.config.model_id,
                model.config.compute_location.value,
                status,
                active
            )
        
        return table
    
    def check_all_models(self):
        """Check availability of all models"""
        console.print(f"\n[bold]Checking availability of {len(self.models)} models...[/bold]")
        
        for name, model in self.models.items():
            with console.status(f"Checking {name}..."):
                if isinstance(model, PlaceholderModel):
                    status = "🔧 Needs Setup"
                else:
                    try:
                        available = model.is_available()
                        status = "🟢 Available" if available else "🔴 Unavailable"
                    except Exception as e:
                        console.print(f"[yellow]Error checking {name}: {e}[/yellow]")
                        status = "❓ Unknown"
                console.print(f"{name}: {status}")
    
    def setup_phi3_ollama(self) -> bool:
        """Setup Phi-3 via Ollama"""
        console.print("\n[bold]Setting up Phi-3 via Ollama...[/bold]")
        
        # Check if we have a Phi-3 Ollama model configured
        phi_model = None
        for model in self.models.values():
            if isinstance(model, PhiOllamaModel):
                phi_model = model
                break
        
        if not phi_model:
            console.print("[red]No Phi-3 Ollama model configured[/red]")
            return False
        
        # Check if Ollama is running
        if not phi_model.is_available():
            console.print("[yellow]Ollama not available or model not installed[/yellow]")
            
            # Try to pull the model
            if Confirm.ask("Pull Phi-3 model from Ollama?"):
                return phi_model.pull_model()
            return False
        
        console.print("[green]✓ Phi-3 via Ollama is ready[/green]")
        return True
    
    def setup_phi3_vllm(self) -> bool:
        """Setup Phi-3 via vLLM"""
        console.print("\n[bold]Setting up Phi-3 via vLLM...[/bold]")
        
        # Check if vLLM is installed
        try:
            import vllm
            console.print("[green]✓ vLLM is installed[/green]")
        except ImportError:
            console.print("[yellow]vLLM not installed[/yellow]")
            if Confirm.ask("Install vLLM?"):
                if not VLLMManager.install_vllm():
                    return False
            else:
                return False
        
        # Find vLLM model
        vllm_model = None
        for model in self.models.values():
            if isinstance(model, PhiVLLMModel):
                vllm_model = model
                break
        
        if not vllm_model:
            console.print("[red]No Phi-3 vLLM model configured[/red]")
            return False
        
        # Start server if local
        if vllm_model.config.compute_location == ComputeLocation.LOCAL:
            console.print("[cyan]Starting local vLLM server...[/cyan]")
            return vllm_model.start_server()
        else:
            # Check remote server
            if vllm_model.is_available():
                console.print("[green]✓ Remote vLLM server is available[/green]")
                return True
            else:
                console.print("[red]Remote vLLM server not available[/red]")
                return False
    
    def interactive_setup(self):
        """Interactive model setup wizard"""
        console.print(Panel.fit(
            "[bold green]Corpus Models Setup Wizard[/bold green]\n"
            "Configure your AI models for Corpus CLI\n\n"
            "[yellow]Note: API keys are managed through 'corpus config setup'[/yellow]",
            border_style="green"
        ))
        
        while True:
            console.print("\n[bold]Setup Options:[/bold]")
            console.print("1. Check API keys and refresh models")
            console.print("2. Setup Phi-3 via Ollama")
            console.print("3. Setup Phi-3 via vLLM")
            console.print("4. Set active model")
            console.print("5. View all models")
            console.print("6. Exit")
            
            choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4", "5", "6"])
            
            if choice == "1":
                self._check_and_configure_api_keys()
            elif choice == "2":
                self.setup_phi3_ollama()
            elif choice == "3":
                self.setup_phi3_vllm()
            elif choice == "4":
                self._set_active_model_interactive()
            elif choice == "5":
                console.print(self.list_models())
            elif choice == "6":
                break
    
    def _check_and_configure_api_keys(self):
        """Check and configure API keys through config manager"""
        console.print("\n[cyan]Checking API Key Configuration[/cyan]")
        
        openai_key = self._get_api_key("openai")
        gemini_key = self._get_api_key("gemini")
        
        console.print(f"OpenAI API Key: {'✓ Configured' if openai_key else '✗ Missing'}")
        console.print(f"Gemini API Key: {'✓ Configured' if gemini_key else '✗ Missing'}")
        
        if not openai_key or not gemini_key:
            console.print("\n[yellow]Missing API keys detected.[/yellow]")
            console.print("Please run: [cyan]corpus config setup[/cyan]")
            console.print("Or set them manually:")
            console.print("  [dim]export OPENAI_API_KEY='your-key-here'[/dim]")
            console.print("  [dim]export GEMINI_API_KEY='your-key-here'[/dim]")
        
        # Refresh models with current keys
        self.refresh_models()
    
    def _set_active_model_interactive(self):
        """Interactive active model selection"""
        if not self.models:
            console.print("[red]No models configured[/red]")
            return
        
        console.print("\n[bold]Available models:[/bold]")
        model_names = list(self.models.keys())
        for i, name in enumerate(model_names, 1):
            model = self.models[name]
            if isinstance(model, PlaceholderModel):
                status = "🔧"
            else:
                status = "🟢" if model.is_available() else "🔴"
            active = " (active)" if name == self.active_model else ""
            console.print(f"{i}. {status} {name}{active}")
        
        try:
            choice = int(Prompt.ask("Select model number")) - 1
            if 0 <= choice < len(model_names):
                selected_model = self.models[model_names[choice]]
                if isinstance(selected_model, PlaceholderModel):
                    console.print("[yellow]This model needs to be configured first[/yellow]")
                    console.print("Run: [cyan]corpus config setup[/cyan] to configure API keys")
                else:
                    self.set_active_model(model_names[choice])
            else:
                console.print("[red]Invalid choice[/red]")
        except ValueError:
            console.print("[red]Invalid input[/red]")