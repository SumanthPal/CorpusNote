# Enhanced diagram generation methods with ChromaDB integration

import re
import json
import asyncio
from typing import Optional, Dict, List, Tuple, Union, Any
from pathlib import Path
import subprocess
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
import chromadb
import google.generativeai as genai
from functools import lru_cache
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError


class DiagramType(Enum):
    """Supported diagram types with descriptions"""
    FLOWCHART = ("flowchart", "Process flows, decision trees, workflows")
    NETWORK = ("network", "System architecture, network diagrams, infrastructure")
    HIERARCHY = ("hierarchy", "Organizational charts, tree structures, taxonomies")
    SEQUENCE = ("sequence", "Interaction diagrams, communication flows, timelines")
    MIND_MAP = ("mind_map", "Concept maps, brainstorming, idea relationships")
    SYSTEM = ("system", "System components and relationships")
    ERD = ("erd", "Entity relationship diagrams for databases")
    STATE = ("state", "State machines and transitions")
    GANTT = ("gantt", "Project timelines and dependencies")
    AUTO = ("auto", "Let AI choose the best type")


class LayoutEngine(Enum):
    """D2 layout engines with descriptions"""
    DAGRE = ("dagre", "Good for flowcharts and hierarchical diagrams")
    ELK = ("elk", "Better for complex network diagrams")
    TALA = ("tala", "Optimized for large graphs (requires D2 Pro)")
    # Note: 'auto' is not a valid D2 layout, using dagre as default


@dataclass
class DiagramConfig:
    """Configuration for diagram generation"""
    max_retries: int = 3
    timeout_seconds: int = 30
    default_format: str = "svg"
    default_layout: LayoutEngine = LayoutEngine.DAGRE
    enhanced_styling: bool = True
    validate_syntax: bool = True
    cleanup_temp_files: bool = True
    max_context_length: int = 4000
    temperature: float = 0.6
    top_p: float = 0.9
    max_tokens: int = 2048


@dataclass
class DocumentSource:
    """Represents a document source used in diagram generation"""
    document: str
    page: Union[str, int]
    chunk: int
    content_type: str
    relevance: float
    content_preview: Optional[str] = None


@dataclass
class DiagramResult:
    """Result of diagram generation"""
    success: bool
    output_path: Optional[Path] = None
    error_message: Optional[str] = None
    sources: List[DocumentSource] = field(default_factory=list)
    d2_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DiagramValidator:
    """Validates D2 diagram syntax and structure"""
    
    @staticmethod
    def validate_syntax(d2_code: str) -> Tuple[bool, Optional[str]]:
        """Comprehensive D2 syntax validation"""
        if not d2_code or not d2_code.strip():
            return False, "Empty diagram code"
        
        # Check balanced braces
        if d2_code.count('{') != d2_code.count('}'):
            return False, f"Unbalanced braces: {d2_code.count('{')} opening, {d2_code.count('}')} closing"
        
        # Check balanced brackets
        if d2_code.count('[') != d2_code.count(']'):
            return False, f"Unbalanced brackets: {d2_code.count('[')} opening, {d2_code.count(']')} closing"
        
        # Check for unclosed strings
        lines = d2_code.split('\n')
        for i, line in enumerate(lines):
            if line.count('"') % 2 != 0:
                return False, f"Unclosed string on line {i+1}"
        
        # Check for valid node definitions
        node_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_-]*(?:\.[a-zA-Z_][a-zA-Z0-9_-]*)*', re.MULTILINE)
        if not node_pattern.search(d2_code):
            return False, "No valid node definitions found"
        
        return True, None
    
    @staticmethod
    def validate_connections(d2_code: str) -> Tuple[bool, Optional[str]]:
        """Validate connection syntax"""
        connection_pattern = re.compile(r'(\w+)\s*-+>\s*(\w+)')
        connections = connection_pattern.findall(d2_code)
        
        if not connections and '->' in d2_code:
            return False, "Invalid connection syntax detected"
        
        return True, None


class DiagramStyler:
    """Handles diagram styling and theming"""
    
    THEMES = {
        "default": {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "accent": "#F18F01",
            "text": "#ffffff",
            "background": "#1a1a1a"
        },
        "professional": {
            "primary": "#1e3a8a",
            "secondary": "#3b82f6",
            "accent": "#60a5fa",
            "text": "#1f2937",
            "background": "#f9fafb"
        },
        "vibrant": {
            "primary": "#dc2626",
            "secondary": "#f59e0b",
            "accent": "#10b981",
            "text": "#ffffff",
            "background": "#18181b"
        }
    }
    
    @classmethod
    def apply_theme(cls, d2_code: str, theme: str = "default") -> str:
        """Apply a theme to D2 code"""
        if theme not in cls.THEMES:
            theme = "default"
        
        colors = cls.THEMES[theme]
        
        style_header = f"""# Theme: {theme}
vars: {{
  primary-color: "{colors['primary']}"
  secondary-color: "{colors['secondary']}"
  accent-color: "{colors['accent']}"
  text-color: "{colors['text']}"
  bg-color: "{colors['background']}"
}}

# Default styles
classes: {{
  important: {{
    style.fill: var(accent-color)
    style.stroke: var(primary-color)
    style.stroke-width: 3
  }}
  
  container: {{
    style.fill: var(bg-color)
    style.stroke: var(secondary-color)
    style.opacity: 0.9
  }}
}}

"""
        return style_header + d2_code


class DiagramGenerator:
    """Enhanced diagram generation with ChromaDB integration and better D2 support"""

    def __init__(self, 
                 diagrams_path: str, 
                 d2_available: bool, 
                 console: Console = None,
                 db_client: chromadb.PersistentClient = None, 
                 collection_name: str = None,
                 config: DiagramConfig = None):
        
        self.diagrams_path = Path(diagrams_path)
        self.d2_available = d2_available
        self.console = console or Console()
        self.config = config or DiagramConfig()
        self.validator = DiagramValidator()
        self.styler = DiagramStyler()
        
        # Create diagrams directory
        self.diagrams_path.mkdir(parents=True, exist_ok=True)
        
        # ChromaDB integration
        self.db_client = db_client
        self.collection_name = collection_name
        self.collection = None
        self.collection_exists = False
        
        self._initialize_chromadb()
        self._initialize_ai_model()
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB connection with error handling"""
        if self.db_client and self.collection_name:
            try:
                self.collection = self.db_client.get_collection(name=self.collection_name)
                self.collection_exists = True
                self.console.print("[dim]📊 Diagram generator connected to document database[/dim]")
            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not connect to document collection: {e}[/yellow]")
    
    def _initialize_ai_model(self):
        """Initialize Gemini AI model with error handling"""
        self.model = None
        try:
            from .config import GEMINI_API_KEY, GEMINI_MODEL
            if GEMINI_API_KEY:
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel(GEMINI_MODEL)
                self.console.print("[dim]🤖 AI model initialized successfully[/dim]")
        except ImportError:
            self.console.print("[yellow]Warning: Gemini not configured. Limited diagram generation capabilities.[/yellow]")
        except Exception as e:
            self.console.print(f"[yellow]Warning: AI model initialization failed: {e}[/yellow]")
    
    def check_collection(self) -> bool:
        """Check if collection exists and has documents"""
        if not self.collection_exists:
            return False
        
        try:
            count = self.collection.count()
            return count > 0
        except Exception:
            return False
    
    def search_documents_for_diagrams(self, 
                                      query: str, 
                                      diagram_type: str = None,
                                      n_results: int = 5) -> Tuple[str, List[DocumentSource]]:
        """Search documents for diagram-relevant content with caching"""
        if not self.check_collection():
            return "", []
        
        try:
            # Don't use Progress here to avoid conflicts
            if hasattr(self.console, 'print'):
                self.console.print("[dim]🔍 Searching documents for diagram context...[/dim]")
            
            # Enhance query for diagram-specific search
            type_keywords = {
                "flowchart": "process workflow steps procedure flow",
                "network": "architecture system infrastructure components connections",
                "hierarchy": "organization structure levels tree parent child",
                "sequence": "interaction communication messages timeline order",
                "erd": "entity relationship database table foreign key",
                "state": "states transitions conditions events"
            }
            
            enhanced_query = query
            if diagram_type and diagram_type in type_keywords:
                enhanced_query += f" {type_keywords[diagram_type]}"
            
            # Search for relevant content
            results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=n_results
            )
            
            # Process results
            sources = []
            context_parts = []
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results.get('distances', [[0] * len(results['documents'][0])])[0]
            )):
                # Create source object
                source = DocumentSource(
                    document=metadata.get('source', 'Unknown'),
                    page=metadata.get('page', 'Unknown'),
                    chunk=metadata.get('chunk_index', 0) + 1,
                    content_type=metadata.get('content_type', 'text'),
                    relevance=1 - distance,
                    content_preview=doc[:200] + "..." if len(doc) > 200 else doc
                )
                sources.append(source)
                
                # Format context
                context_part = f"[Source: {source.document} | Page: {source.page}]\n{doc}\n"
                context_parts.append(context_part)
            
            # Limit context length
            full_context = "\n---\n".join(context_parts)
            if len(full_context) > self.config.max_context_length:
                full_context = full_context[:self.config.max_context_length] + "\n[Context truncated...]"
            
            return full_context, sources
            
        except Exception as e:
            self.console.print(f"[red]Document search error: {e}[/red]")
            return "", []
    
    def generate_diagram_from_query(self, 
                                    query: str, 
                                    diagram_type: Union[str, DiagramType] = DiagramType.AUTO,
                                    use_context: bool = True, 
                                    layout: Union[str, LayoutEngine] = LayoutEngine.DAGRE,
                                    theme: str = "default") -> DiagramResult:
        """Generate diagram from natural language query with comprehensive error handling"""
        
        if not self.model:
            return DiagramResult(
                success=False,
                error_message="AI model not available for diagram generation"
            )
        
        # Convert enums to strings
        if isinstance(diagram_type, DiagramType):
            diagram_type = diagram_type.value[0]
        if isinstance(layout, LayoutEngine):
            layout = layout.value[0]
        
        # Handle 'auto' layout by defaulting to dagre
        if layout == "auto":
            layout = "dagre"
        
        # Search for context if requested
        context = ""
        sources = []
        if use_context and self.collection_exists:
            context, source_objects = self.search_documents_for_diagrams(query, diagram_type)
            sources = source_objects
        
        # Generate diagram with retries
        d2_code = None
        last_error = ""
        
        for attempt in range(self.config.max_retries):
            self.console.print(f"[dim]🎨 Generating diagram code (Attempt {attempt + 1}/{self.config.max_retries})...[/dim]")
            
            try:
                # Generate D2 code with timeout
                future = self.executor.submit(
                    self._generate_d2_code_with_ai,
                    query, diagram_type, context, last_error
                )
                d2_code = future.result(timeout=self.config.timeout_seconds)
                
                if not d2_code:
                    last_error = "AI failed to generate diagram code"
                    continue
                
                # Validate syntax
                if self.config.validate_syntax:
                    valid, error = self.validator.validate_syntax(d2_code)
                    if not valid:
                        last_error = f"Syntax validation failed: {error}"
                        continue
                
                # Apply theme
                if self.config.enhanced_styling:
                    d2_code = self.styler.apply_theme(d2_code, theme)
                
                # Render diagram
                output_file, render_error = self._render_diagram_safe(
                    d2_code, 
                    f"query_diagram_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                    layout
                )
                
                if output_file:
                    return DiagramResult(
                        success=True,
                        output_path=output_file,
                        sources=sources,
                        d2_code=d2_code,
                        metadata={
                            "query": query,
                            "diagram_type": diagram_type,
                            "layout": layout,
                            "theme": theme,
                            "context_used": bool(context),
                            "attempts": attempt + 1
                        }
                    )
                else:
                    last_error = render_error or "Unknown rendering error"
                    
            except TimeoutError:
                last_error = "Diagram generation timed out"
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
        
        return DiagramResult(
            success=False,
            error_message=f"Failed after {self.config.max_retries} attempts. Last error: {last_error}",
            sources=sources
        )
    
    def _generate_d2_code_with_ai(self, 
                                  query: str, 
                                  diagram_type: str, 
                                  context: str, 
                                  error_feedback: str = "") -> Optional[str]:
        """Generate D2 diagram code using AI with improved prompting"""
        
        # Build structured prompt
        prompt_parts = [
            "You are an expert D2 diagram generator. Generate ONLY valid D2 code without any explanations or markdown.",
            f"\nTASK: Create a {diagram_type} diagram for: {query}",
            "\nD2 SYNTAX RULES:",
            "- Nodes: `nodeName` or `\"Node Name with Spaces\"`",
            "- Connections: `A -> B` or `A -> B: \"Label\"`",
            "- Containers: `container { child1; child2 }`",
            "- Shapes: `node.shape: rectangle` (rectangle, circle, diamond, cylinder, cloud, hexagon)",
            "- Styles: `node.style.fill: \"#color\"` and `node.style.stroke: \"#color\"`",
            "- Classes: `node.class: className`",
            "- Ensure ALL braces {} are properly matched",
            "- Use meaningful node names and labels"
        ]
        
        # Add diagram-specific guidance
        diagram_guidance = {
            "flowchart": "\nFLOWCHART SPECIFIC:\n- Use diamond shapes for decisions\n- Rectangle for processes\n- Circle for start/end\n- Clear directional flow",
            "network": "\nNETWORK SPECIFIC:\n- Use containers for network segments\n- Show connections between components\n- Include labels for protocols/ports",
            "hierarchy": "\nHIERARCHY SPECIFIC:\n- Use containers for grouping\n- Show parent-child relationships clearly\n- Maintain consistent levels",
            "sequence": "\nSEQUENCE SPECIFIC:\n- Show actors/participants\n- Use arrows for messages\n- Order interactions chronologically",
            "erd": "\nERD SPECIFIC:\n- Use containers for entities\n- Show attributes inside entities\n- Label relationships clearly",
            "state": "\nSTATE SPECIFIC:\n- Use circles for states\n- Label transitions with conditions\n- Mark initial and final states"
        }
        
        if diagram_type in diagram_guidance:
            prompt_parts.append(diagram_guidance[diagram_type])
        
        # Add context if available
        if context:
            prompt_parts.append(f"\nCONTEXT FROM DOCUMENTS:\n{context}")
        
        # Add error feedback for retries
        if error_feedback:
            prompt_parts.append(f"\nPREVIOUS ATTEMPT FAILED:\n{error_feedback}\nPlease fix the issues and generate corrected D2 code.")
        
        prompt_parts.append("\nGenerate the D2 code now:")
        prompt = "\n".join(prompt_parts)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_output_tokens=self.config.max_tokens,
                )
            )
            
            # Extract and clean D2 code
            d2_code = response.text.strip()
            
            # Remove common artifacts
            d2_code = re.sub(r'^```d2\s*', '', d2_code, flags=re.MULTILINE)
            d2_code = re.sub(r'```\s*$', '', d2_code)
            d2_code = re.sub(r'^```\s*', '', d2_code, flags=re.MULTILINE)
            
            return d2_code.strip()
            
        except Exception as e:
            self.console.print(f"[red]AI generation error: {e}[/red]")
            return None
    
    def _render_diagram_safe(self, 
                             d2_code: str, 
                             filename: str,
                             layout: str = "dagre") -> Tuple[Optional[Path], Optional[str]]:
        """Safely render D2 diagram with error handling"""
        if not self.d2_available:
            return None, "D2 tool not available"
        
        # Validate layout engine
        valid_layouts = ["dagre", "elk", "tala"]
        if layout not in valid_layouts:
            layout = "dagre"  # Default to dagre if invalid
        
        output_file = self.diagrams_path / f"{filename}.{self.config.default_format}"
        
        # Use temporary file for D2 code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.d2', delete=False) as tmp:
            tmp.write(d2_code)
            tmp_path = Path(tmp.name)
        
        try:
            # Run D2 command
            cmd = ['d2', '-l', layout, str(tmp_path), str(output_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.config.timeout_seconds)
            
            if result.returncode == 0:
                return output_file, None
            else:
                return None, f"D2 rendering failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return None, "D2 rendering timed out"
        except Exception as e:
            return None, f"Rendering error: {str(e)}"
        finally:
            # Cleanup temp file
            if self.config.cleanup_temp_files and tmp_path.exists():
                tmp_path.unlink()
    
    def create_diagram_from_documents(self, 
                                      search_query: str, 
                                      diagram_type: Union[str, DiagramType] = DiagramType.AUTO,
                                      max_documents: int = 10,
                                      theme: str = "default") -> DiagramResult:
        """Create diagram specifically from document search results"""
        
        if not self.check_collection():
            return DiagramResult(
                success=False,
                error_message="No document collection available"
            )
        
        # Convert enum to string
        if isinstance(diagram_type, DiagramType):
            diagram_type = diagram_type.value[0]
        
        # Search for documents
        context, sources = self.search_documents_for_diagrams(
            search_query, diagram_type, max_documents
        )
        
        if not context:
            return DiagramResult(
                success=False,
                error_message="No relevant documents found"
            )
        
        # Generate focused query
        diagram_query = (
            f"Create a comprehensive {diagram_type} diagram based on the provided documents. "
            f"Focus on: {search_query}. "
            "Extract key concepts, relationships, and processes from the documents."
        )
        
        # Generate diagram
        result = self.generate_diagram_from_query(
            diagram_query,
            diagram_type,
            use_context=True,
            theme=theme
        )
        
        # Add document-specific metadata
        if result.success:
            result.metadata['source_documents'] = len(sources)
            result.metadata['search_query'] = search_query
        
        return result
    
    def batch_generate_diagrams(self, 
                                queries: List[Dict[str, Any]]) -> List[DiagramResult]:
        """Generate multiple diagrams in parallel"""
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task(f"Generating {len(queries)} diagrams...", total=len(queries))
            
            # Process in parallel
            futures = []
            for query_config in queries:
                future = self.executor.submit(
                    self.generate_diagram_from_query,
                    **query_config
                )
                futures.append(future)
            
            # Collect results
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=self.config.timeout_seconds * 2)
                    results.append(result)
                except Exception as e:
                    results.append(DiagramResult(
                        success=False,
                        error_message=f"Batch generation failed: {str(e)}"
                    ))
                progress.update(task, advance=1)
        
        return results
    
    def export_diagram(self, 
                       diagram_path: Path, 
                       export_format: str,
                       output_path: Optional[Path] = None) -> Optional[Path]:
        """Export diagram to different formats"""
        if not diagram_path.exists():
            self.console.print(f"[red]Diagram not found: {diagram_path}[/red]")
            return None
        
        if not output_path:
            output_path = diagram_path.with_suffix(f'.{export_format}')
        
        try:
            if diagram_path.suffix == '.d2':
                # Re-render from D2 source
                d2_code = diagram_path.read_text()
                cmd = ['d2', str(diagram_path), str(output_path)]
            else:
                # Convert existing diagram
                if export_format == diagram_path.suffix[1:]:
                    shutil.copy(diagram_path, output_path)
                    return output_path
                
                # Use ImageMagick or similar for conversion
                cmd = ['convert', str(diagram_path), str(output_path)]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.console.print(f"[green]✅ Exported to: {output_path}[/green]")
                return output_path
            else:
                self.console.print(f"[red]Export failed: {result.stderr}[/red]")
                return None
                
        except Exception as e:
            self.console.print(f"[red]Export error: {e}[/red]")
            return None
    
    def get_diagram_info(self, diagram_path: Path) -> Optional[Dict[str, Any]]:
        """Get detailed information about a diagram"""
        if not diagram_path.exists():
            return None
        
        stat = diagram_path.stat()
        info = {
            "filename": diagram_path.name,
            "path": str(diagram_path),
            "format": diagram_path.suffix[1:],
            "size_bytes": stat.st_size,
            "size_human": self._format_size(stat.st_size),
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
        }
        
        # Try to read D2 source if available
        d2_path = diagram_path.with_suffix('.d2')
        if d2_path.exists():
            info["has_source"] = True
            info["source_path"] = str(d2_path)
            try:
                d2_code = d2_path.read_text()
                info["nodes"] = len(re.findall(r'^\s*(\w+)[:\s{]', d2_code, re.MULTILINE))
                info["connections"] = len(re.findall(r'->', d2_code))
            except:
                pass
        
        return info
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def list_available_diagrams(self, 
                                 sort_by: str = "modified",
                                 filter_format: Optional[str] = None) -> List[Path]:
        """List all generated diagrams with sorting and filtering"""
        if not self.diagrams_path.exists():
            return []
        
        # Get all diagram files
        patterns = ['*.svg', '*.png', '*.pdf'] if not filter_format else [f'*.{filter_format}']
        diagram_files = []
        for pattern in patterns:
            diagram_files.extend(self.diagrams_path.glob(pattern))
        
        # Sort based on criteria
        if sort_by == "modified":
            diagram_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        elif sort_by == "created":
            diagram_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        elif sort_by == "name":
            diagram_files.sort(key=lambda x: x.name)
        elif sort_by == "size":
            diagram_files.sort(key=lambda x: x.stat().st_size, reverse=True)
        
        return diagram_files
    
    def show_diagram_gallery(self, 
                             limit: int = 20,
                             show_preview: bool = False):
        """Show gallery of generated diagrams with enhanced display"""
        diagrams = self.list_available_diagrams()[:limit]
        
        if not diagrams:
            self.console.print("[yellow]No diagrams found[/yellow]")
            return
        
        # Create summary stats
        total_size = sum(d.stat().st_size for d in diagrams)
        formats = {}
        for d in diagrams:
            fmt = d.suffix[1:].upper()
            formats[fmt] = formats.get(fmt, 0) + 1
        
        # Show summary
        summary = f"[bold]Diagram Gallery[/bold]\n"
        summary += f"Total diagrams: {len(diagrams)}\n"
        summary += f"Total size: {self._format_size(total_size)}\n"
        summary += f"Formats: {', '.join(f'{fmt} ({count})' for fmt, count in formats.items())}"
        
        self.console.print(Panel(summary, title="Gallery Summary", border_style="blue"))
        
        # Show diagram table
        table = Table(title="Generated Diagrams", show_lines=True)
        table.add_column("Filename", style="cyan", no_wrap=True)
        table.add_column("Format", style="magenta")
        table.add_column("Size", style="green", justify="right")
        table.add_column("Modified", style="yellow")
        table.add_column("Info", style="dim")
        
        for diagram in diagrams:
            info = self.get_diagram_info(diagram)
            
            # Build info string
            info_parts = []
            if info.get("has_source"):
                info_parts.append("📄 Source")
            if info.get("nodes"):
                info_parts.append(f"🔵 {info['nodes']} nodes")
            if info.get("connections"):
                info_parts.append(f"➡️ {info['connections']} connections")
            
            table.add_row(
                diagram.name,
                info['format'].upper(),
                info['size_human'],
                info['modified'].strftime('%Y-%m-%d %H:%M'),
                " ".join(info_parts)
            )
        
        self.console.print(table)
        
        # Show preview option
        if show_preview and diagrams:
            self.console.print("\n[dim]Use 'view <filename>' to preview a diagram[/dim]")
    
    def suggest_diagram_types(self, query: str) -> List[Tuple[DiagramType, float]]:
        """Suggest appropriate diagram types based on query with confidence scores"""
        query_lower = query.lower()
        suggestions = []
        
        # Define keywords for each diagram type with weights
        type_keywords = {
            DiagramType.FLOWCHART: {
                'keywords': ['process', 'workflow', 'steps', 'procedure', 'flow', 'algorithm', 'decision'],
                'weight': 1.0
            },
            DiagramType.NETWORK: {
                'keywords': ['network', 'architecture', 'system', 'infrastructure', 'server', 'connection', 'topology'],
                'weight': 1.0
            },
            DiagramType.HIERARCHY: {
                'keywords': ['hierarchy', 'organization', 'structure', 'tree', 'levels', 'parent', 'child', 'org chart'],
                'weight': 1.0
            },
            DiagramType.SEQUENCE: {
                'keywords': ['sequence', 'interaction', 'communication', 'messages', 'timeline', 'protocol', 'events'],
                'weight': 1.0
            },
            DiagramType.ERD: {
                'keywords': ['entity', 'relationship', 'database', 'table', 'schema', 'foreign key', 'model'],
                'weight': 1.2
            },
            DiagramType.STATE: {
                'keywords': ['state', 'machine', 'transition', 'status', 'lifecycle', 'fsm'],
                'weight': 1.1
            },
            DiagramType.MIND_MAP: {
                'keywords': ['mind map', 'concept', 'brainstorm', 'ideas', 'thoughts', 'planning'],
                'weight': 0.9
            },
            DiagramType.GANTT: {
                'keywords': ['gantt', 'timeline', 'project', 'schedule', 'milestone', 'deadline'],
                'weight': 1.1
            }
        }
        
        # Calculate scores for each type
        for diagram_type, config in type_keywords.items():
            score = 0.0
            for keyword in config['keywords']:
                if keyword in query_lower:
                    # Give higher score for exact matches
                    if f' {keyword} ' in f' {query_lower} ':
                        score += 2.0 * config['weight']
                    else:
                        score += 1.0 * config['weight']
            
            if score > 0:
                suggestions.append((diagram_type, score))
        
        # Sort by score and normalize
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        # If no matches, provide default suggestions
        if not suggestions:
            suggestions = [
                (DiagramType.FLOWCHART, 0.5),
                (DiagramType.NETWORK, 0.3),
                (DiagramType.HIERARCHY, 0.2)
            ]
        
        # Normalize scores to confidence (0-1)
        if suggestions:
            max_score = suggestions[0][1]
            if max_score > 0:
                suggestions = [(dt, score/max_score) for dt, score in suggestions]
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def interactive_diagram_generator(self):
        """Enhanced interactive diagram generation interface"""
        # Show welcome message
        welcome_text = """
[bold green]Interactive Diagram Generator[/bold green]
Generate beautiful diagrams from natural language or document searches.

[bold]Quick Commands:[/bold]
• [cyan]create[/cyan] - Generate diagram from description
• [cyan]search[/cyan] - Create diagram from documents
• [cyan]batch[/cyan] - Generate multiple diagrams
• [cyan]gallery[/cyan] - Browse generated diagrams
• [cyan]export[/cyan] - Convert diagram format
• [cyan]help[/cyan] - Show detailed help
• [cyan]exit[/cyan] - Exit the generator
        """
        self.console.print(Panel(welcome_text, title="Welcome", border_style="green"))
        
        # Show system status
        self._show_system_status()
        
        # Command history
        command_history = []
        
        while True:
            try:
                # Get user input
                self.console.print()
                command = self.console.input("[bold blue]diagram>[/bold blue] ").strip()
                
                if not command:
                    continue
                
                command_history.append(command)
                
                # Parse command
                parts = command.split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                # Handle commands
                if cmd in ['exit', 'quit', 'q']:
                    self._handle_exit()
                    break
                
                elif cmd == 'help':
                    self._show_detailed_help()
                
                elif cmd == 'create':
                    self._handle_create_command_interactive(args)
                
                elif cmd == 'search':
                    self._handle_search_command_interactive(args)
                
                elif cmd == 'batch':
                    self._handle_batch_command()
                
                elif cmd == 'gallery':
                    self.show_diagram_gallery(show_preview=True)
                
                elif cmd == 'export':
                    self._handle_export_command(args)
                
                elif cmd == 'view':
                    self._handle_view_command(args)
                
                elif cmd == 'types':
                    self._show_diagram_types()
                
                elif cmd == 'themes':
                    self._show_themes()
                
                elif cmd == 'status':
                    self._show_system_status()
                
                elif cmd == 'history':
                    self._show_command_history(command_history)
                
                else:
                    # Try to interpret as a direct creation command
                    if len(command) > 10:
                        self.console.print("[dim]Interpreting as diagram description...[/dim]")
                        self._handle_create_command_interactive(command)
                    else:
                        self.console.print(f"[yellow]Unknown command: '{cmd}'. Type 'help' for available commands.[/yellow]")
            
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                import traceback
                if self.console.is_terminal:
                    traceback.print_exc()
    
    def _show_system_status(self):
        """Show system status and capabilities"""
        status_items = []
        
        # D2 availability
        if self.d2_available:
            status_items.append("✅ D2 renderer available")
        else:
            status_items.append("❌ D2 renderer not found")
        
        # AI model
        if self.model:
            status_items.append("✅ AI model connected")
        else:
            status_items.append("❌ AI model not available")
        
        # Document database
        if self.collection_exists:
            try:
                count = self.collection.count()
                status_items.append(f"✅ Document database ({count} chunks)")
            except:
                status_items.append("⚠️ Document database (error)")
        else:
            status_items.append("❌ Document database not connected")
        
        # Diagram count
        diagram_count = len(self.list_available_diagrams())
        status_items.append(f"📊 {diagram_count} diagrams generated")
        
        self.console.print(Panel(
            "\n".join(status_items),
            title="System Status",
            border_style="blue"
        ))
    
    def _handle_create_command_interactive(self, description: str):
        """Handle interactive diagram creation"""
        if not description:
            description = self.console.input("📝 Describe your diagram: ").strip()
            if not description:
                self.console.print("[yellow]Description required[/yellow]")
                return
        
        # Get suggestions
        suggestions = self.suggest_diagram_types(description)
        
        # Show suggestions
        self.console.print("\n[bold]Suggested diagram types:[/bold]")
        for i, (dtype, confidence) in enumerate(suggestions[:3]):
            conf_bar = "🟩" * int(confidence * 5) + "⬜" * (5 - int(confidence * 5))
            self.console.print(f"  {i+1}. {dtype.value[0]} {conf_bar} ({confidence:.0%})")
        
        # Get user choice
        choice = self.console.input("\nSelect type (1-3) or enter custom [default=1]: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= 3:
            diagram_type = suggestions[int(choice)-1][0].value[0]
        elif choice:
            diagram_type = choice
        else:
            diagram_type = suggestions[0][0].value[0]
        
        # Get theme
        theme = self.console.input("Theme (default/professional/vibrant) [default]: ").strip() or "default"
        
        # Get layout
        layout = self.console.input("Layout (dagre/elk) [dagre]: ").strip() or "dagre"
        if layout not in ["dagre", "elk"]:
            layout = "dagre"
        
        # Generate with status message (no Progress widget)
        self.console.print("\n[bold green]Generating diagram...[/bold green]")
        result = self.generate_diagram_from_query(
            description,
            diagram_type=diagram_type,
            use_context=True,
            layout=layout,
            theme=theme
        )
        
        # Show result
        if result.success:
            self.console.print(f"\n[green]✅ Diagram generated successfully![/green]")
            self.console.print(f"📁 Saved to: [cyan]{result.output_path}[/cyan]")
            
            # Show preview if possible
            if result.d2_code:
                self.console.print("\n[bold]D2 Code Preview:[/bold]")
                syntax = Syntax(result.d2_code[:500] + "..." if len(result.d2_code) > 500 else result.d2_code,
                                "d2", theme="monokai", line_numbers=True)
                self.console.print(syntax)
        else:
            self.console.print(f"\n[red]❌ Generation failed: {result.error_message}[/red]")
    
    def _handle_search_command_interactive(self, query: str):
        """Handle interactive document search diagram creation"""
        if not self.collection_exists:
            self.console.print("[yellow]No document database available[/yellow]")
            return
        
        if not query:
            query = self.console.input("🔍 Search query: ").strip()
            if not query:
                self.console.print("[yellow]Search query required[/yellow]")
                return
        
        # Preview search results
        self.console.print("\n[dim]Searching documents...[/dim]")
        _, sources = self.search_documents_for_diagrams(query, n_results=3)
        
        if sources:
            self.console.print(f"\n[green]Found {len(sources)} relevant documents[/green]")
            for i, source in enumerate(sources[:3]):
                self.console.print(f"  {i+1}. {source.document} (Page {source.page}) - {source.relevance:.0%} relevant")
        else:
            self.console.print("[yellow]No relevant documents found[/yellow]")
            return
        
        # Continue with diagram generation
        if self.console.input("\nGenerate diagram from these sources? [Y/n]: ").strip().lower() != 'n':
            # Get diagram type
            suggestions = self.suggest_diagram_types(query)
            diagram_type = suggestions[0][0].value[0] if suggestions else "auto"
            
            type_choice = self.console.input(f"Diagram type [{diagram_type}]: ").strip() or diagram_type
            
            # Generate
            with self.console.status("[bold green]Creating diagram from documents...", spinner="dots"):
                result = self.create_diagram_from_documents(
                    query,
                    diagram_type=type_choice,
                    max_documents=10
                )
            
            # Show result
            if result.success:
                self.console.print(f"\n[green]✅ Diagram created from {len(result.sources)} documents![/green]")
                self.console.print(f"📁 Saved to: [cyan]{result.output_path}[/cyan]")
                
                # Show source summary
                if result.sources:
                    table = Table(title="Document Sources Used")
                    table.add_column("Document", style="cyan")
                    table.add_column("Relevance", style="yellow")
                    
                    for source in result.sources[:5]:
                        table.add_row(
                            f"{source.document} (p.{source.page})",
                            f"{source.relevance:.0%}"
                        )
                    
                    self.console.print(table)
            else:
                self.console.print(f"\n[red]❌ Generation failed: {result.error_message}[/red]")
    
    def _handle_batch_command(self):
        """Handle batch diagram generation"""
        self.console.print("\n[bold]Batch Diagram Generation[/bold]")
        self.console.print("Enter diagram descriptions (one per line, empty line to finish):")
        
        queries = []
        while True:
            desc = self.console.input(f"  {len(queries)+1}. ").strip()
            if not desc:
                break
            
            # Parse description for type hints
            if ":" in desc:
                dtype, desc = desc.split(":", 1)
                dtype = dtype.strip()
                desc = desc.strip()
            else:
                suggestions = self.suggest_diagram_types(desc)
                dtype = suggestions[0][0].value[0] if suggestions else "auto"
            
            queries.append({
                "query": desc,
                "diagram_type": dtype,
                "use_context": True
            })
        
        if not queries:
            self.console.print("[yellow]No diagrams to generate[/yellow]")
            return
        
        # Confirm
        self.console.print(f"\n[dim]Ready to generate {len(queries)} diagrams[/dim]")
        if self.console.input("Continue? [Y/n]: ").strip().lower() == 'n':
            return
        
        # Generate
        results = self.batch_generate_diagrams(queries)
        
        # Show results
        success_count = sum(1 for r in results if r.success)
        self.console.print(f"\n[bold]Batch Results:[/bold]")
        self.console.print(f"✅ Success: {success_count}/{len(results)}")
        
        if success_count < len(results):
            self.console.print(f"❌ Failed: {len(results) - success_count}")
            for i, result in enumerate(results):
                if not result.success:
                    self.console.print(f"   - Diagram {i+1}: {result.error_message}")
    
    def _handle_export_command(self, args: str):
        """Handle diagram export"""
        parts = args.split()
        if len(parts) < 2:
            self.console.print("[yellow]Usage: export <filename> <format>[/yellow]")
            self.console.print("Example: export query_diagram_20240101_120000.svg png")
            return
        
        filename = parts[0]
        export_format = parts[1].lower()
        
        # Find the diagram
        diagram_path = self.diagrams_path / filename
        if not diagram_path.exists():
            # Try to find by partial match
            matches = list(self.diagrams_path.glob(f"*{filename}*"))
            if matches:
                diagram_path = matches[0]
                self.console.print(f"[dim]Found: {diagram_path.name}[/dim]")
            else:
                self.console.print(f"[red]Diagram not found: {filename}[/red]")
                return
        
        # Export
        output_path = self.export_diagram(diagram_path, export_format)
        if output_path:
            self.console.print(f"[green]✅ Exported to: {output_path}[/green]")
    
    def _handle_view_command(self, filename: str):
        """Handle diagram viewing (show metadata)"""
        if not filename:
            self.console.print("[yellow]Usage: view <filename>[/yellow]")
            return
        
        # Find the diagram
        diagram_path = self.diagrams_path / filename
        if not diagram_path.exists():
            matches = list(self.diagrams_path.glob(f"*{filename}*"))
            if matches:
                diagram_path = matches[0]
            else:
                self.console.print(f"[red]Diagram not found: {filename}[/red]")
                return
        
        # Get and show info
        info = self.get_diagram_info(diagram_path)
        if info:
            info_text = f"""
[bold]Diagram Information[/bold]
📁 File: {info['filename']}
📊 Format: {info['format'].upper()}
💾 Size: {info['size_human']}
📅 Created: {info['created'].strftime('%Y-%m-%d %H:%M:%S')}
📝 Modified: {info['modified'].strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            if info.get('has_source'):
                info_text += f"\n📄 Source available: {info['source_path']}"
                if info.get('nodes'):
                    info_text += f"\n🔵 Nodes: {info['nodes']}"
                if info.get('connections'):
                    info_text += f"\n➡️ Connections: {info['connections']}"
            
            self.console.print(Panel(info_text.strip(), title=f"Diagram: {info['filename']}", border_style="blue"))
            
            # Show D2 code preview if available
            if info.get('has_source'):
                if self.console.input("\nShow D2 source code? [y/N]: ").strip().lower() == 'y':
                    d2_code = Path(info['source_path']).read_text()
                    syntax = Syntax(d2_code, "d2", theme="monokai", line_numbers=True)
                    self.console.print("\n[bold]D2 Source Code:[/bold]")
                    self.console.print(syntax)
    
    def _show_diagram_types(self):
        """Show available diagram types with descriptions"""
        table = Table(title="Available Diagram Types", show_lines=True)
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Description", style="green")
        table.add_column("Best For", style="yellow")
        
        for dtype in DiagramType:
            if dtype == DiagramType.AUTO:
                continue
            
            best_for = {
                DiagramType.FLOWCHART: "Processes, algorithms, decision flows",
                DiagramType.NETWORK: "System architecture, infrastructure",
                DiagramType.HIERARCHY: "Org charts, taxonomies, trees",
                DiagramType.SEQUENCE: "API calls, protocols, interactions",
                DiagramType.ERD: "Database schemas, data models",
                DiagramType.STATE: "State machines, lifecycle diagrams",
                DiagramType.MIND_MAP: "Brainstorming, concept mapping",
                DiagramType.GANTT: "Project timelines, schedules"
            }
            
            table.add_row(
                dtype.value[0],
                dtype.value[1],
                best_for.get(dtype, "General purpose")
            )
        
        self.console.print(table)
    
    def _show_themes(self):
        """Show available themes"""
        table = Table(title="Available Themes", show_lines=True)
        table.add_column("Theme", style="cyan")
        table.add_column("Primary", style="blue")
        table.add_column("Secondary", style="magenta")
        table.add_column("Accent", style="yellow")
        table.add_column("Description", style="green")
        
        descriptions = {
            "default": "Balanced colors for general use",
            "professional": "Clean, corporate appearance",
            "vibrant": "Bold, eye-catching colors"
        }
        
        for theme_name, colors in DiagramStyler.THEMES.items():
            table.add_row(
                theme_name,
                f"[{colors['primary']}]████[/]",
                f"[{colors['secondary']}]████[/]",
                f"[{colors['accent']}]████[/]",
                descriptions.get(theme_name, "Custom theme")
            )
        
        self.console.print(table)
    
    def _show_command_history(self, history: List[str]):
        """Show command history"""
        if not history:
            self.console.print("[yellow]No command history[/yellow]")
            return
        
        self.console.print("\n[bold]Command History:[/bold]")
        for i, cmd in enumerate(history[-10:], 1):
            self.console.print(f"  {i:2d}. {cmd}")
    
    def _show_detailed_help(self):
        """Show detailed help information"""
        help_sections = {
            "Basic Commands": {
                "create <description>": "Generate a diagram from natural language",
                "search <query>": "Create diagram from document search",
                "gallery": "Browse all generated diagrams",
                "types": "Show available diagram types",
                "themes": "Show available color themes",
                "exit": "Exit the generator"
            },
            "Advanced Commands": {
                "batch": "Generate multiple diagrams at once",
                "export <file> <format>": "Convert diagram to different format",
                "view <file>": "Show detailed diagram information",
                "status": "Show system status and capabilities",
                "history": "Show command history"
            },
            "Quick Tips": {
                "Direct creation": "Just type a description to create a diagram",
                "Type hints": "Use 'type:description' format (e.g., 'network:aws architecture')",
                "Context": "Diagrams can use your document database for accuracy",
                "Themes": "Try different themes for various visual styles"
            }
        }
        
        for section, commands in help_sections.items():
            self.console.print(f"\n[bold]{section}:[/bold]")
            for cmd, desc in commands.items():
                self.console.print(f"  [cyan]{cmd}[/cyan] - {desc}")
    
    def _handle_exit(self):
        """Handle graceful exit"""
        # Show session summary
        diagram_count = len(self.list_available_diagrams())
        self.console.print(f"\n[dim]Generated {diagram_count} diagrams this session[/dim]")
        self.console.print("[yellow]Thanks for using Diagram Generator! Goodbye! 👋[/yellow]")


# Example usage and integration
if __name__ == "__main__":
    # This would typically be integrated with your main application
    console = Console()
    
    # Initialize with mock configuration
    generator = DiagramGenerator(
        diagrams_path="./diagrams",
        d2_available=True,
        console=console,
        config=DiagramConfig(
            max_retries=3,
            enhanced_styling=True,
            default_format="svg"
        )
    )
    
    # Run interactive mode
    generator.interactive_diagram_generator()