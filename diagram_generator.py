# Enhanced diagram generation methods with ChromaDB integration

import re
import json
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import subprocess
from datetime import datetime
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.markdown import Markdown
import chromadb
import google.generativeai as genai

class DiagramGenerator:
    """Enhanced diagram generation with ChromaDB integration and better D2 support"""

    def __init__(self, diagrams_path: str, d2_available: bool, console: Console = None,
                 db_client: chromadb.PersistentClient = None, collection_name: str = None):
        self.diagrams_path = Path(diagrams_path)
        self.d2_available = d2_available
        self.console = console or Console()

        # ChromaDB integration
        self.db_client = db_client
        self.collection_name = collection_name
        self.collection = None
        self.collection_exists = False

        # Initialize ChromaDB connection if provided
        if self.db_client and self.collection_name:
            try:
                self.collection = self.db_client.get_collection(name=self.collection_name)
                self.collection_exists = True
                if hasattr(self.console, 'print'):
                    self.console.print("[dim]📊 Diagram generator connected to document database[/dim]")
            except Exception as e:
                if hasattr(self.console, 'print'):
                    self.console.print(f"[yellow]Warning: Could not connect to document collection: {e}[/yellow]")

        # Initialize Gemini for diagram generation
        self.model = None
        try:
            # Assume GEMINI_API_KEY is available from config
            from config import GEMINI_API_KEY, GEMINI_MODEL
            if GEMINI_API_KEY:
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel(GEMINI_MODEL)
        except ImportError:
            if hasattr(self.console, 'print'):
                self.console.print("[yellow]Warning: Gemini not configured. Limited diagram generation capabilities.[/yellow]")

        self.layout_engines = {
            'dagre': 'Good for flowcharts and hierarchical diagrams',
            'elk': 'Better for complex network diagrams',
            'tala': 'Optimized for large graphs',
            'auto': 'Let D2 choose the best layout'
        }

    def check_collection(self) -> bool:
        """Check if collection exists and has documents"""
        if not self.collection_exists:
            if hasattr(self.console, 'print'):
                self.console.print("[yellow]No document collection available for context-aware diagrams[/yellow]")
            return False

        try:
            count = self.collection.count()
            if count == 0:
                if hasattr(self.console, 'print'):
                    self.console.print("[yellow]No documents in database for context[/yellow]")
                return False
            return True
        except Exception as e:
            if hasattr(self.console, 'print'):
                self.console.print(f"[yellow]Error checking collection: {e}[/yellow]")
            return False

    def search_documents_for_diagrams(self, query: str, diagram_type: str = None,
                                      n_results: int = 5) -> Tuple[str, List[Dict]]:
        """Search documents for diagram-relevant content"""
        if not self.check_collection():
            return "", []

        try:
            # Enhance query for diagram-specific search
            enhanced_query = query
            if diagram_type:
                enhanced_query += f" {diagram_type} diagram flow process structure"

            # Search for relevant content
            results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=n_results
            )

            # Format context
            context_parts = []
            sources = []

            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results.get('distances', [[0] * len(results['documents'][0])])[0]
            )):
                source_info = {
                    "document": metadata.get('source', 'Unknown'),
                    "page": metadata.get('page', 'Unknown'),
                    "chunk": metadata.get('chunk_index', 0) + 1,
                    "content_type": metadata.get('content_type', 'text'),
                    "relevance": 1 - distance
                }
                sources.append(source_info)

                # Format context for diagram generation
                context_part = f"[{source_info['document']} | {source_info['page']}]\n{doc}\n"
                context_parts.append(context_part)

            full_context = "\n---\n".join(context_parts)
            return full_context, sources

        except Exception as e:
            if hasattr(self.console, 'print'):
                self.console.print(f"[red]Document search error: {e}[/red]")
            return "", []

    def generate_diagram_from_query(self, query: str, diagram_type: str = 'auto',
                                    use_context: bool = True, layout: str = 'dagre', max_retries: int = 3) -> Optional[Path]:
        """Generate diagram from natural language query with optional document context"""

        if not self.model:
            if hasattr(self.console, 'print'):
                self.console.print("[red]❌ Gemini model not available for diagram generation[/red]")
            return None

        # Search for relevant document context if requested
        context = ""
        sources = []
        if use_context and self.collection_exists:
            if hasattr(self.console, 'print'):
                self.console.print("[dim]🔍 Searching documents for diagram context...[/dim]")
            context, sources = self.search_documents_for_diagrams(query, diagram_type)

        d2_code = None
        last_error = ""

        for attempt in range(max_retries):
            if hasattr(self.console, 'print'):
                self.console.print(f"[dim]🎨 Generating diagram code (Attempt {attempt + 1}/{max_retries})...[/dim]")
            
            d2_code = self._generate_d2_code_with_ai(query, diagram_type, context, last_error)

            if not d2_code:
                if hasattr(self.console, 'print'):
                    self.console.print("[red]❌ AI failed to generate diagram code.[/red]")
                last_error = "AI failed to produce any D2 code. Please try again."
                continue

            # Render the diagram to validate it
            output_file, error = self.render_diagram(d2_code, f"query_diagram_validation", layout, return_error=True)

            if output_file:
                # Success
                if hasattr(self.console, 'print'):
                    summary_text = f"[green]✅ Diagram generated from query![/green]\n"
                    summary_text += f"📝 Query: {query[:60]}{'...' if len(query) > 60 else ''}\n"
                    summary_text += f"📊 Type: {diagram_type}\n"
                    summary_text += f"🎯 Layout: {layout}\n"
                    if sources:
                        summary_text += f"📚 Context: {len(sources)} document chunks\n"
                    summary_text += f"💾 Saved to: [cyan]{output_file}[/cyan]"
                    self.console.print(Panel.fit(summary_text, title="Diagram Generation Complete", border_style="green"))
                
                # Clean up validation file
                if output_file.exists():
                     output_file.unlink()
                     d2_file = output_file.with_suffix('.d2')
                     if d2_file.exists():
                        d2_file.unlink()

                # Final render with proper name
                final_output = self.render_diagram(d2_code, "query_diagram", layout)
                return final_output
            else:
                # Failure, prepare for retry
                last_error = error
                if hasattr(self.console, 'print'):
                    self.console.print(f"[yellow]⚠️ Attempt {attempt + 1} failed. Retrying with error feedback...[/yellow]")
        
        if hasattr(self.console, 'print'):
            self.console.print(f"[red]❌ Failed to generate a valid diagram after {max_retries} attempts.[/red]")
            self.console.print(f"[red]Last error: {last_error}[/red]")

        return None

    def _generate_d2_code_with_ai(self, query: str, diagram_type: str, context: str, error_feedback: str = "") -> str:
        """Generate D2 diagram code using AI with document context and error feedback"""

        # Build comprehensive prompt
        prompt = f"""You are an expert diagram generator. Your task is to create valid D2 diagramming language code.

TASK: Generate D2 diagram code for the following request: "{query}"
DIAGRAM TYPE: {diagram_type}

D2 SYNTAX RULES:
- Connections: `A -> B: "Label"`
- Shapes: `node.shape: rectangle` (valid shapes: rectangle, circle, diamond, cylinder, etc.)
- Styling: `node.style.fill: "#C0FFEE"`
- Containers: `container: {{ child1; child2 }}` (ensure braces are matched)
- Do NOT add any text or comments outside of the D2 code block itself.
- Ensure all strings are properly quoted and terminated.

"""
        # Add context if available
        if context:
            prompt += f"""DOCUMENT CONTEXT (use this to make the diagram accurate):
{context}
"""
        # Add error feedback if this is a retry
        if error_feedback:
            prompt += f"""CORRECTION REQUEST:
The previous attempt failed with the following error. Please fix the D2 code.
ERROR: {error_feedback}

Analyze the error and the previous code, then generate a new, corrected D2 code block.
"""

        prompt += """
REQUIREMENTS:
1.  Generate ONLY valid D2 code. Do not include any explanations, markdown blocks (` ```d2 `), or other text.
2.  Use appropriate shapes, labels, and connections.
3.  Ensure all braces `{}` are correctly matched and balanced.

Generate the D2 code now:"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.6,
                    top_p=0.9,
                    max_output_tokens=2048,
                )
            )
            # Robustly extract D2 code
            d2_code = response.text.strip()
            
            # More aggressive cleaning of common AI artifacts
            d2_code = re.sub(r'^```d2\s*', '', d2_code, flags=re.MULTILINE)
            d2_code = re.sub(r'```\s*$', '', d2_code)
            
            return d2_code.strip()

        except Exception as e:
            if hasattr(self.console, 'print'):
                self.console.print(f"[red]AI generation error: {e}[/red]")
            return None


    def create_diagram_from_documents(self, search_query: str, diagram_type: str = 'auto',
                                      max_documents: int = 10) -> Optional[Path]:
        """Create diagram specifically from document search results"""

        if not self.check_collection():
            return None

        # Search for relevant documents
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("📚 Searching documents...", total=None)

            context, sources = self.search_documents_for_diagrams(
                search_query, diagram_type, max_documents
            )

            if not context:
                if hasattr(self.console, 'print'):
                    self.console.print("[yellow]No relevant documents found for diagram creation[/yellow]")
                return None

            progress.update(task, description="🎨 Generating diagram from documents...")

            # Create diagram instruction based on document content
            diagram_query = f"Create a {diagram_type} diagram showing the key concepts, relationships, and processes described in the provided documents about: {search_query}"

            output_file = self.generate_diagram_from_query(diagram_query, diagram_type, use_context=True)
            
            if output_file:
                progress.update(task, description="✅ Diagram completed!")

                # Show sources used
                if hasattr(self.console, 'print'):
                    sources_table = self._create_sources_table(sources)
                    self.console.print("\n")
                    self.console.print(sources_table)

        return output_file

    def _create_sources_table(self, sources: List[Dict]) -> Table:
        """Create a table showing document sources used for diagram"""
        table = Table(title="Document Sources Used", show_lines=True)
        table.add_column("Document", style="cyan")
        table.add_column("Page", style="green")
        table.add_column("Type", style="magenta")
        table.add_column("Relevance", justify="right", style="yellow")

        for source in sources:
            content_type = source.get('content_type', 'text')
            type_display = "📷 Image" if content_type == 'image' else "📄 Text"

            relevance_bar = "▰" * int(source['relevance'] * 10) + "▱" * (10 - int(source['relevance'] * 10))

            table.add_row(
                source['document'],
                str(source['page']),
                type_display,
                f"{source['relevance']:.2f} {relevance_bar}"
            )

        return table

    def list_available_diagrams(self) -> List[Path]:
        """List all generated diagrams"""
        if not self.diagrams_path.exists():
            return []

        diagram_files = []
        for ext in ['*.svg', '*.png', '*.pdf']:
            diagram_files.extend(self.diagrams_path.glob(ext))

        return sorted(diagram_files, key=lambda x: x.stat().st_mtime, reverse=True)

    def show_diagram_gallery(self):
        """Show gallery of generated diagrams"""
        diagrams = self.list_available_diagrams()

        if not diagrams:
            if hasattr(self.console, 'print'):
                self.console.print("[yellow]No diagrams found[/yellow]")
            return

        table = Table(title="Generated Diagrams", show_lines=True)
        table.add_column("Filename", style="cyan")
        table.add_column("Format", style="magenta")
        table.add_column("Size", style="green")
        table.add_column("Created", style="yellow")

        for diagram in diagrams:
            file_size = diagram.stat().st_size
            size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
            created = datetime.fromtimestamp(diagram.stat().st_mtime).strftime('%Y-%m-%d %H:%M')

            table.add_row(
                diagram.name,
                diagram.suffix.upper()[1:],
                size_str,
                created
            )

        if hasattr(self.console, 'print'):
            self.console.print(table)

    def suggest_diagram_types(self, query: str) -> List[str]:
        """Suggest appropriate diagram types based on query"""
        query_lower = query.lower()
        suggestions = []

        # Keyword-based suggestions
        if any(word in query_lower for word in ['process', 'workflow', 'steps', 'procedure']):
            suggestions.append('flowchart')

        if any(word in query_lower for word in ['network', 'architecture', 'system', 'infrastructure']):
            suggestions.append('network')

        if any(word in query_lower for word in ['hierarchy', 'organization', 'structure', 'tree']):
            suggestions.append('hierarchy')

        if any(word in query_lower for word in ['sequence', 'interaction', 'communication', 'messages']):
            suggestions.append('sequence')

        if any(word in query_lower for word in ['relationship', 'connection', 'link', 'association']):
            suggestions.append('network')

        if any(word in query_lower for word in ['mind map', 'concept', 'brainstorm', 'ideas']):
            suggestions.append('mind_map')

        # Default suggestions if none match
        if not suggestions:
            suggestions = ['flowchart', 'network', 'hierarchy']

        return list(dict.fromkeys(suggestions)) # Remove duplicates

    def interactive_diagram_generator(self):
        """Interactive diagram generation interface"""
        if hasattr(self.console, 'print'):
            self.console.print(Panel.fit(
                "[bold green]Interactive Diagram Generator[/bold green]\n"
                "Generate diagrams from natural language or document searches.\n\n"
                "Commands:\n"
                "• 'create <description>' - Generate diagram from description\n"
                "• 'search <query>' - Create diagram from document search\n"
                "• 'gallery' - Show generated diagrams\n"
                "• 'types' - Show available diagram types\n"
                "• 'help' - Show this help\n"
                "• 'exit' - Exit generator",
                title="Diagram Generator"
            ))

        # Show database status
        if self.collection_exists:
            count = self.collection.count()
            if hasattr(self.console, 'print'):
                self.console.print(f"[dim]📚 Connected to database with {count} document chunks[/dim]")

        while True:
            try:
                if hasattr(self.console, 'print'):
                    self.console.print()
                    command = self.console.input("[bold]Diagram>[/bold] ").strip()
                else:
                    command = input("Diagram> ").strip()

                if command.lower() in ['exit', 'quit', 'q']:
                    if hasattr(self.console, 'print'):
                        self.console.print("[yellow]Goodbye![/yellow]")
                    break

                elif command.lower() == 'gallery':
                    self.show_diagram_gallery()

                elif command.lower() == 'types':
                    self._show_diagram_types()

                elif command.lower() == 'help':
                    self._show_help()

                elif command.lower().startswith('create '):
                    description = command[7:].strip()
                    if description:
                        self._handle_create_command(description)
                    else:
                        if hasattr(self.console, 'print'):
                            self.console.print("[yellow]Please provide a description[/yellow]")

                elif command.lower().startswith('search '):
                    query = command[7:].strip()
                    if query:
                        self._handle_search_command(query)
                    else:
                        if hasattr(self.console, 'print'):
                            self.console.print("[yellow]Please provide a search query[/yellow]")

                elif command.strip() == '':
                    continue

                else:
                    if hasattr(self.console, 'print'):
                        self.console.print("[yellow]Unknown command. Type 'help' for available commands.[/yellow]")

            except KeyboardInterrupt:
                if hasattr(self.console, 'print'):
                    self.console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except Exception as e:
                if hasattr(self.console, 'print'):
                    self.console.print(f"[red]Error: {e}[/red]")

    def _handle_create_command(self, description: str):
        """Handle diagram creation from description"""
        # Suggest diagram types
        suggestions = self.suggest_diagram_types(description)

        if hasattr(self.console, 'print'):
            self.console.print(f"[dim]💡 Suggested diagram types: {', '.join(suggestions)}[/dim]")
            diagram_type = self.console.input(f"Diagram type (default: {suggestions[0]}): ").strip() or suggestions[0]
        else:
            diagram_type = suggestions[0]

        # Generate diagram
        self.generate_diagram_from_query(description, diagram_type)


    def _handle_search_command(self, query: str):
        """Handle diagram creation from document search"""
        if not self.collection_exists:
            if hasattr(self.console, 'print'):
                self.console.print("[yellow]No document database available for search[/yellow]")
            return

        # Suggest diagram types
        suggestions = self.suggest_diagram_types(query)

        if hasattr(self.console, 'print'):
            self.console.print(f"[dim]💡 Suggested diagram types: {', '.join(suggestions)}[/dim]")
            diagram_type = self.console.input(f"Diagram type (default: {suggestions[0]}): ").strip() or suggestions[0]
        else:
            diagram_type = suggestions[0]

        # Create diagram from documents
        self.create_diagram_from_documents(query, diagram_type)


    def _show_diagram_types(self):
        """Show available diagram types"""
        types_info = {
            'flowchart': 'Process flows, decision trees, workflows',
            'network': 'System architecture, network diagrams, infrastructure',
            'hierarchy': 'Organizational charts, tree structures, taxonomies',
            'sequence': 'Interaction diagrams, communication flows, timelines',
            'mind_map': 'Concept maps, brainstorming, idea relationships',
            'system': 'System components and relationships',
            'auto': 'Let AI choose the best type'
        }

        table = Table(title="Available Diagram Types")
        table.add_column("Type", style="cyan")
        table.add_column("Description", style="green")

        for dtype, description in types_info.items():
            table.add_row(dtype, description)

        if hasattr(self.console, 'print'):
            self.console.print(table)

    def _show_help(self):
        """Show help information"""
        help_text = """
[bold]Diagram Generator Commands:[/bold]

[cyan]create <description>[/cyan]
  Generate a diagram from natural language description
  Example: create user login process flowchart

[cyan]search <query>[/cyan]
  Create diagram from document search results
  Example: search network architecture

[cyan]gallery[/cyan]
  Show all generated diagrams

[cyan]types[/cyan]
  Show available diagram types

[cyan]help[/cyan]
  Show this help message

[cyan]exit[/cyan]
  Exit the diagram generator

[bold]Tips:[/bold]
• Be specific in your descriptions
• Use keywords like "process", "network", "hierarchy" for better type suggestions
• The generator uses document context when available
• Generated diagrams are saved in the diagrams directory
"""
        if hasattr(self.console, 'print'):
            self.console.print(Panel(help_text, title="Help"))

    def validate_d2_syntax(self, d2_code: str) -> bool:
        """Basic validation of D2 syntax. Returns True if likely valid."""
        if d2_code.count('{') != d2_code.count('}'):
            return False
        if d2_code.count('[') != d2_code.count(']'):
            return False
        # Add more sophisticated checks if needed
        return True


    def enhance_d2_code(self, d2_code: str, diagram_type: str = 'auto') -> str:
        """Enhance D2 code with better styling and structure"""
        enhanced_code = d2_code

        if 'style:' not in enhanced_code and 'fill:' not in enhanced_code:
            style_header = """
# Enhanced styling
vars: {
  primary-color: "#2E86AB"
  secondary-color: "#A23B72"
  accent-color: "#F18F01"
  text-color: "#ffffff"
}

"""
            enhanced_code = style_header + enhanced_code

        return enhanced_code

    def render_diagram(self, d2_code: str, filename_prefix: str = "diagram",
                       layout: str = 'dagre', format: str = 'svg', return_error: bool = False) -> Tuple[Optional[Path], Optional[str]]:
        """Render D2 code, returning the path on success or an error message on failure."""
        if not self.d2_available:
            msg = "D2 tool not available. Install from [https://d2lang.com/tour/install](https://d2lang.com/tour/install)"
            if hasattr(self.console, 'print'):
                self.console.print(f"[red]❌ {msg}[/red]")
            return (None, msg) if return_error else None

        enhanced_code = self.enhance_d2_code(d2_code)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{filename_prefix}_{timestamp}"
        d2_file = self.diagrams_path / f"{base_name}.d2"
        output_file = self.diagrams_path / f"{base_name}.{format}"

        try:
            self.diagrams_path.mkdir(parents=True, exist_ok=True)
            d2_file.write_text(enhanced_code, encoding='utf-8')

            command = ['d2', '-l', layout, str(d2_file), str(output_file)]
            
            result = subprocess.run(command, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                error_msg = f"D2 rendering failed: {result.stderr}"
                return (None, error_msg) if return_error else None
            
            return (output_file, None) if return_error else output_file

        except Exception as e:
            error_msg = f"Unexpected error during rendering: {e}"
            return (None, error_msg) if return_error else None
        finally:
            # Clean up the temporary .d2 file
            if d2_file.exists():
                pass # Keep the D2 file for debugging if needed, or unlink it: d2_file.unlink()

