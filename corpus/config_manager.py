#!/usr/bin/env python3
"""Configuration management for Corpus CLI"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.panel import Panel
from dotenv import load_dotenv, set_key, dotenv_values
import typer

# Load environment variables
load_dotenv()

console = Console()

class ConfigManager:
    """Manages configuration for Corpus CLI"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".corpus"
        self.config_file = self.config_dir / "config.json"
        self.env_file = Path(".env")
        self.config_dir.mkdir(exist_ok=True)
        
        # Default configuration
        self.defaults = {
            # API Keys (from env)
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", os.getenv("GEMINI", "")),
            
            # Models
            "GEMINI_MODEL": "gemini-1.5-flash",
            "GEMINI_IMG_MODEL": "gemini-pro-vision",
            
            # Paths
            "DB_PATH": str(self.config_dir / "research.db"),
            "DIAGRAMS_PATH": str(self.config_dir / "diagrams"),
            "COLLECTION_NAME": "documents",
            
            # Processing
            "CHUNK_SIZE": 1000,
            "CHUNK_OVERLAP": 200,
            "MIN_CHUNK_LENGTH": 50,
            "MAX_FILE_SIZE_MB": 100,
            
            # Search
            "MAX_RESULTS": 5,
            "MAX_MEMORY": 10,
            
            # File Extensions
            "CODE_EXTENSIONS": [
                '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
                '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
                '.r', '.m', '.cs', '.sh', '.bash', '.zsh', '.fish',
                '.sql', '.jsx', '.tsx', '.vue', '.yaml', '.yml', '.json',
                '.xml', '.html', '.css', '.scss', '.less',
                '.dockerfile', '.makefile', '.cmake', '.gradle'
            ],
            "IMAGE_EXTENSIONS": ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp', '.svg'],
            "TEXT_EXTENSIONS": ['.txt', '.md', '.pdf', '.docx', '.rtf'],
            
            # Ignore patterns
            "IGNORE_DIRECTORIES": [
                "__pycache__", "node_modules", ".git", ".idea", ".vscode",
                "venv", ".venv", "env", "dist", "build", "eggs", ".eggs"
            ],
            "IGNORE_FILES": [
                ".DS_Store", "*.log", "*.tmp", "*.swp", "*.swo", "thumbs.db"
            ]
        }
        
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create with defaults"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults (defaults for any missing keys)
                config = self.defaults.copy()
                config.update(loaded_config)
                return config
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load config: {e}[/yellow]")
        
        # Create default config
        self.save_config(self.defaults)
        return self.defaults.copy()
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Save configuration to file"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            console.print(f"[green]✓ Configuration saved to {self.config_file}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving config: {e}[/red]")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        # Check environment first for API keys
        if key == "GEMINI_API_KEY":
            env_value = os.getenv("GEMINI_API_KEY", os.getenv("GEMINI"))
            if env_value:
                return env_value
        
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value
        self.save_config()
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values"""
        self.config.update(updates)
        self.save_config()
    
    def reset(self, key: Optional[str] = None) -> None:
        """Reset configuration to defaults"""
        if key:
            if key in self.defaults:
                self.config[key] = self.defaults[key]
                console.print(f"[green]✓ Reset {key} to default[/green]")
            else:
                console.print(f"[red]Unknown configuration key: {key}[/red]")
        else:
            self.config = self.defaults.copy()
            console.print("[green]✓ Reset all configuration to defaults[/green]")
        self.save_config()
    
    def show(self, key: Optional[str] = None, show_all: bool = False) -> None:
        """Display configuration"""
        if key:
            value = self.get(key)
            if value is not None:
                console.print(f"[cyan]{key}[/cyan] = {json.dumps(value, indent=2)}")
            else:
                console.print(f"[red]Unknown configuration key: {key}[/red]")
            return
        
        # Show all configuration
        table = Table(title="Corpus Configuration", show_lines=True)
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Type", style="yellow")
        
        # Group settings
        groups = {
            "API Keys": ["GEMINI_API_KEY"],
            "Models": ["GEMINI_MODEL", "GEMINI_IMG_MODEL"],
            "Paths": ["DB_PATH", "DIAGRAMS_PATH", "COLLECTION_NAME"],
            "Processing": ["CHUNK_SIZE", "CHUNK_OVERLAP", "MIN_CHUNK_LENGTH", "MAX_FILE_SIZE_MB"],
            "Search": ["MAX_RESULTS", "MAX_MEMORY"]
        }
        
        for group_name, keys in groups.items():
            table.add_row(f"[bold]{group_name}[/bold]", "", "")
            for key in keys:
                value = self.get(key)
                if key == "GEMINI_API_KEY" and value and not show_all:
                    display_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                else:
                    display_value = str(value)
                table.add_row(f"  {key}", display_value, type(value).__name__)
        
        # File extensions (summary)
        if not show_all:
            table.add_row("[bold]File Types[/bold]", "", "")
            table.add_row("  CODE_EXTENSIONS", f"{len(self.get('CODE_EXTENSIONS', []))} extensions", "list")
            table.add_row("  IMAGE_EXTENSIONS", f"{len(self.get('IMAGE_EXTENSIONS', []))} extensions", "list")
            table.add_row("  TEXT_EXTENSIONS", f"{len(self.get('TEXT_EXTENSIONS', []))} extensions", "list")
        
        console.print(table)
        
        if not show_all:
            console.print("\n[dim]Use 'corpus config show --all' to see full lists[/dim]")
    
    def interactive_setup(self) -> None:
        """Interactive configuration setup"""
        console.print(Panel.fit(
            "[bold]Corpus Configuration Setup[/bold]\n"
            "Let's configure your Corpus CLI installation.",
            border_style="green"
        ))
        
        updates = {}
        
        # API Key
        console.print("\n[bold]1. API Configuration[/bold]")
        current_key = self.get("GEMINI_API_KEY")
        if current_key:
            console.print(f"Current Gemini API key: {current_key[:8]}...{current_key[-4:]}")
            if Confirm.ask("Update Gemini API key?", default=False):
                key = Prompt.ask("Enter your Gemini API key", password=True)
                updates["GEMINI_API_KEY"] = key
                # Also save to .env file
                set_key(self.env_file, "GEMINI_API_KEY", key)
        else:
            console.print("[yellow]No Gemini API key found[/yellow]")
            key = Prompt.ask("Enter your Gemini API key", password=True)
            updates["GEMINI_API_KEY"] = key
            set_key(self.env_file, "GEMINI_API_KEY", key)
        
        # Model selection
        console.print("\n[bold]2. Model Selection[/bold]")
        models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
        current_model = self.get("GEMINI_MODEL")
        console.print(f"Current model: {current_model}")
        if Confirm.ask("Change model?", default=False):
            for i, model in enumerate(models, 1):
                console.print(f"  {i}. {model}")
            choice = IntPrompt.ask("Select model", default=1, choices=[str(i) for i in range(1, len(models)+1)])
            updates["GEMINI_MODEL"] = models[int(choice)-1]
        
        # Paths
        console.print("\n[bold]3. Storage Paths[/bold]")
        if Confirm.ask("Configure storage paths?", default=False):
            db_path = Prompt.ask("Database path", default=self.get("DB_PATH"))
            updates["DB_PATH"] = db_path
            
            diagram_path = Prompt.ask("Diagrams path", default=self.get("DIAGRAMS_PATH"))
            updates["DIAGRAMS_PATH"] = diagram_path
        
        # Processing parameters
        console.print("\n[bold]4. Processing Parameters[/bold]")
        if Confirm.ask("Configure processing parameters?", default=False):
            chunk_size = IntPrompt.ask("Chunk size", default=self.get("CHUNK_SIZE"))
            updates["CHUNK_SIZE"] = chunk_size
            
            chunk_overlap = IntPrompt.ask("Chunk overlap", default=self.get("CHUNK_OVERLAP"))
            updates["CHUNK_OVERLAP"] = chunk_overlap
            
            max_file_size = IntPrompt.ask("Max file size (MB)", default=self.get("MAX_FILE_SIZE_MB"))
            updates["MAX_FILE_SIZE_MB"] = max_file_size
        
        # Apply updates
        if updates:
            self.update(updates)
            console.print("\n[green]✓ Configuration updated successfully![/green]")
        else:
            console.print("\n[yellow]No changes made[/yellow]")
    
    def export_config(self, format: str = "json") -> str:
        """Export configuration to string"""
        if format == "yaml":
            return yaml.dump(self.config, default_flow_style=False)
        elif format == "env":
            lines = []
            for key, value in self.config.items():
                if isinstance(value, (str, int, float)):
                    lines.append(f"{key}={value}")
            return "\n".join(lines)
        else:  # json
            return json.dumps(self.config, indent=2)
    
    def import_config(self, data: str, format: str = "json") -> bool:
        """Import configuration from string"""
        try:
            if format == "yaml":
                new_config = yaml.safe_load(data)
            elif format == "env":
                new_config = {}
                for line in data.strip().split('\n'):
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        new_config[key.strip()] = value.strip()
            else:  # json
                new_config = json.loads(data)
            
            self.update(new_config)
            return True
        except Exception as e:
            console.print(f"[red]Error importing config: {e}[/red]")
            return False
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check API key
        if not self.get("GEMINI_API_KEY"):
            issues.append("GEMINI_API_KEY is not set")
        
        # Check paths exist
        db_path = Path(self.get("DB_PATH"))
        if not db_path.parent.exists():
            issues.append(f"Database directory does not exist: {db_path.parent}")
        
        diagram_path = Path(self.get("DIAGRAMS_PATH"))
        if not diagram_path.exists():
            try:
                diagram_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create diagrams directory: {e}")
        
        # Check numeric values
        if self.get("CHUNK_SIZE", 0) < 100:
            issues.append("CHUNK_SIZE should be at least 100")
        
        if self.get("CHUNK_OVERLAP", 0) >= self.get("CHUNK_SIZE", 1):
            issues.append("CHUNK_OVERLAP should be less than CHUNK_SIZE")
        
        return issues


# Global config instance
_config = None

def get_config() -> ConfigManager:
    """Get or create global config instance"""
    global _config
    if _config is None:
        _config = ConfigManager()
    return _config


# Convenience functions for backward compatibility
def get(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return get_config().get(key, default)


def set(key: str, value: Any) -> None:
    """Set configuration value"""
    get_config().set(key, value)


# Export commonly used values as module-level variables
# These are dynamically loaded from config
def __getattr__(name):
    """Dynamic attribute access for config values"""
    config = get_config()
    if name in config.config:
        return config.get(name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# CLI commands for configuration management
def create_config_commands(app: typer.Typer):
    """Create configuration subcommands"""
    
    config_app = typer.Typer(help="Manage Corpus configuration")
    
    @config_app.command("show")
    def show_config(
        key: Optional[str] = typer.Argument(None, help="Specific key to show"),
        all: bool = typer.Option(False, "--all", "-a", help="Show all values including lists")
    ):
        """Show current configuration"""
        config = get_config()
        config.show(key, show_all=all)
    
    @config_app.command("set")
    def set_config(
        key: str = typer.Argument(..., help="Configuration key"),
        value: str = typer.Argument(..., help="Configuration value")
    ):
        """Set a configuration value"""
        config = get_config()
        
        # Try to parse value as appropriate type
        try:
            # Try as JSON first (for lists, etc)
            parsed_value = json.loads(value)
        except:
            # Try as int
            try:
                parsed_value = int(value)
            except:
                # Try as float
                try:
                    parsed_value = float(value)
                except:
                    # Keep as string
                    parsed_value = value
        
        config.set(key, parsed_value)
        console.print(f"[green]✓ Set {key} = {parsed_value}[/green]")
    
    @config_app.command("reset")
    def reset_config(
        key: Optional[str] = typer.Argument(None, help="Specific key to reset"),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
    ):
        """Reset configuration to defaults"""
        config = get_config()
        
        if key:
            config.reset(key)
        else:
            if not yes and not Confirm.ask("Reset all configuration to defaults?"):
                console.print("[yellow]Cancelled[/yellow]")
                return
            config.reset()
    
    @config_app.command("setup")
    def setup_config():
        """Run interactive configuration setup"""
        config = get_config()
        config.interactive_setup()
    
    @config_app.command("validate")
    def validate_config():
        """Validate current configuration"""
        config = get_config()
        issues = config.validate_config()
        
        if issues:
            console.print("[red]Configuration issues found:[/red]")
            for issue in issues:
                console.print(f"  • {issue}")
        else:
            console.print("[green]✓ Configuration is valid[/green]")
    
    @config_app.command("export")
    def export_config(
        format: str = typer.Option("json", "--format", "-f", help="Export format (json, yaml, env)"),
        output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file")
    ):
        """Export configuration"""
        config = get_config()
        exported = config.export_config(format)
        
        if output:
            Path(output).write_text(exported)
            console.print(f"[green]✓ Exported configuration to {output}[/green]")
        else:
            console.print(exported)
    
    @config_app.command("path")
    def show_config_path():
        """Show configuration file location"""
        config = get_config()
        console.print(f"Configuration file: [cyan]{config.config_file}[/cyan]")
        console.print(f"Configuration directory: [cyan]{config.config_dir}[/cyan]")
    
    app.add_typer(config_app, name="config")


# For testing
if __name__ == "__main__":
    config = get_config()
    config.show()

