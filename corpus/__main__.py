#!/usr/bin/env python3
# main.py
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from typing import Optional, List, Dict
import sys
import os
import chromadb
from datetime import datetime
from corpus.config_manager import create_config_commands, get_config

# Initialize components that are cheap and used everywhere
console = Console()

# --- Lazy Loader for Expensive Components ---
# This class prevents slow components from being loaded until they are needed.
class LazyLoader:
    _parser = None
    _chat_interface = None
    _diagram_generator = None
    _web_parser = None
    _db_client = None

    @classmethod
    def get_db_client(cls):
        if cls._db_client is None:
            try:
                config = get_config()
                db_path = config.get("DB_PATH")
                cls._db_client = chromadb.PersistentClient(path=str(db_path))
            except Exception as e:
                console.print(f"[red]Error initializing ChromaDB: {e}[/red]")
                raise typer.Exit(1)
        return cls._db_client

    @classmethod
    def get_parser(cls):
        if cls._parser is None:
            from corpus.file_parser import FileParser
            # Assuming FileParser might use the DB client internally
            cls._parser = FileParser()
        return cls._parser

    @classmethod
    def get_web_parser(cls):
        if cls._web_parser is None:
            from corpus.web_parser import WebParser
            cls._web_parser = WebParser(cls.get_parser())
        return cls._web_parser

    @classmethod
    def get_chat_interface(cls):
        if cls._chat_interface is None:
            from corpus.chat import ChatInterface
            cls._chat_interface = ChatInterface()
        return cls._chat_interface

    @classmethod
    def get_diagram_generator(cls):
        if cls._diagram_generator is None:
            try:
                from corpus.diagram_generator import DiagramGenerator, DiagramConfig
            except ImportError:
                from diagram_generator import DiagramGenerator, DiagramConfig
            
            config = get_config()
            d2_available = cls._check_d2_available()
            
            diagram_config = DiagramConfig(
                max_retries=3,
                enhanced_styling=True,
                default_format="svg",
                cleanup_temp_files=True
            )
            
            cls._diagram_generator = DiagramGenerator(
                diagrams_path=config.get("DIAGRAMS_PATH"),
                d2_available=d2_available,
                console=console,
                db_client=cls.get_db_client(),
                collection_name=config.get("COLLECTION_NAME"),
                config=diagram_config
            )
        return cls._diagram_generator

        
    @staticmethod
    def _check_d2_available():
        """Check if D2 is installed and available"""
        import subprocess
        try:
            result = subprocess.run(['d2', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

# --- CLI App Definition ---
app = typer.Typer(
    name="research",
    help="Research document chat CLI - Index and chat with your documents using AI",
    add_completion=False
)

# Helper function for pretty headers
def print_header(text: str):
    console.print(Panel.fit(f"[bold cyan]{text}[/bold cyan]", padding=(0, 1)))

@app.command()
def index(
    path: str = typer.Argument(..., help="Path to file or directory to index"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R", help="Recursively index subdirectories"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-indexing of existing files"),
    pattern: Optional[str] = typer.Option(None, "--pattern", "-p", help="File pattern to match (e.g., '*atomic*.pdf')"),
    max_workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Number of threads (default: auto-detect optimal)"),
    disable_threading: bool = typer.Option(False, "--no-threading", help="Disable multi-threading (single-threaded mode)")
):
    """Index documents into the research database with optional multi-threading"""
    print_header("Document Indexing")
    parser = LazyLoader.get_parser()
    path_obj = Path(path).expanduser()
    
    # Handle threading configuration
    if disable_threading:
        workers = 1
        console.print("[dim]Running in single-threaded mode[/dim]")
    elif max_workers is not None:
        if max_workers < 1:
            console.print("[red]Error: --workers must be at least 1[/red]")
            raise typer.Exit(1)
        workers = max_workers
        console.print(f"[dim]Using {workers} worker threads[/dim]")
    else:
        # Auto-detect optimal thread count
        workers = None  # Let the method use its default
        optimal = min(32, (os.cpu_count() or 1) * 4)
        console.print(f"[dim]Auto-detected {optimal} optimal worker threads[/dim]")
    
    try:
        if path_obj.is_file():
            # Single file indexing (no threading needed)
            success, message = parser.index_file(path_obj, force=force)
            if success:
                console.print(f"[green]✓ {message}[/green]")
            else:
                console.print(f"[red]✗ {message}[/red]")
                raise typer.Exit(1)
                
        elif path_obj.is_dir():
            # Directory indexing with threading support
            stats = parser.index_directory(
                path_obj, 
                recursive=recursive, 
                force=force, 
                pattern=pattern,
                max_workers=workers
            )
            
            # Enhanced summary output
            console.print(f"\n[bold]Indexing Complete![/bold]")
            console.print(f"[green]✓ Indexed: {stats['indexed']} files[/green]")
            console.print(f"[yellow]⟳ Skipped: {stats['skipped']} files[/yellow]")
            console.print(f"[red]✗ Failed: {stats['failed']} files[/red]")
            
            # Show performance info if available
            total_processed = stats['indexed'] + stats['skipped'] + stats['failed']
            if total_processed > 0:
                success_rate = (stats['indexed'] + stats['skipped']) / total_processed * 100
                console.print(f"[dim]Success rate: {success_rate:.1f}%[/dim]")
                
        else:
            console.print(f"[red]Path not found: {path}[/red]")
            raise typer.Exit(1)
            
    except KeyboardInterrupt:
        console.print(f"\n[yellow]⚠️  Indexing interrupted by user[/yellow]")
        raise typer.Exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        # Show more detailed error in debug mode
        import traceback
        if os.getenv('DEBUG'):
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)

@app.command()
def index_url(
    url: str = typer.Argument(..., help="URL to index (webpage or GitHub repo)"),
    max_pages: int = typer.Option(50, "--max-pages", "-m", help="Maximum pages to crawl (for websites)"),
    same_domain: bool = typer.Option(True, "--same-domain/--any-domain", help="Only crawl same domain"),
    github_token: Optional[str] = typer.Option(None, "--github-token", "-t", help="GitHub personal access token")
):
    """Index a website or GitHub repository"""
    print_header("Web Indexing")
    web_parser = LazyLoader.get_web_parser()
    if 'github.com' in url:
        console.print(f"[cyan]Detected GitHub repository[/cyan]")
        stats = web_parser.index_github_repo(url, github_token)
        console.print(f"\n[bold]GitHub Indexing Complete![/bold]")
        console.print(f"[green]✓ Indexed: {stats['indexed']} files[/green]")
        console.print(f"[red]✗ Failed: {stats['failed']} files[/red]")
    elif url.startswith(('http://', 'https://')):
        if max_pages == 1:
            success = web_parser.index_webpage(url)
            if success:
                console.print(f"[green]✓ Successfully indexed {url}[/green]")
            else:
                console.print(f"[red]✗ Failed to index {url}[/red]")
                raise typer.Exit(1)
        else:
            console.print(f"[cyan]Crawling website (max {max_pages} pages)[/cyan]")
            stats = web_parser.index_website(url, max_pages, same_domain)
            console.print(f"\n[bold]Website Indexing Complete![/bold]")
            console.print(f"[green]✓ Indexed: {stats['indexed']} pages[/green]")
            console.print(f"[red]✗ Failed: {stats['failed']} pages[/red]")
    else:
        console.print(f"[red]Invalid URL. Must start with http:// or https://[/red]")
        raise typer.Exit(1)

@app.command()
def index_github(
    repo: str = typer.Argument(..., help="GitHub repo (format: owner/repo or full URL)"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="GitHub personal access token for private repos")
):
    """Index a GitHub repository (shorthand command)"""
    print_header("GitHub Repository Indexing")
    web_parser = LazyLoader.get_web_parser()
    if not repo.startswith('http'):
        if '/' in repo:
            repo = f"https://github.com/{repo}"
        else:
            console.print("[red]Invalid format. Use: owner/repo or full GitHub URL[/red]")
            raise typer.Exit(1)
    stats = web_parser.index_github_repo(repo, token)
    console.print(f"\n[bold]Indexing Complete![/bold]")
    console.print(f"[green]✓ Indexed: {stats['indexed']} files[/green]")
    console.print(f"[red]✗ Failed: {stats['failed']} files[/red]")

@app.command()
def chat(
    filter: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter documents by filename pattern"),
    no_sources: bool = typer.Option(False, "--no-sources", help="Hide source citations")
):
    """Start interactive chat with your documents"""
    print_header("Research Assistant Chat")
    chat_interface = LazyLoader.get_chat_interface()
    chat_interface.interactive_chat(document_filter=filter)

@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about your documents"),
    filter: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter documents by filename pattern"),
    no_sources: bool = typer.Option(False, "--no-sources", help="Hide source citations")
):
    """Ask a single question about your documents (non-interactive)"""
    chat_interface = LazyLoader.get_chat_interface()
    chat_interface.ask_single(question, filter_pattern=filter, show_sources=not no_sources)

@app.command()
def diagram(
    query: Optional[str] = typer.Argument(None, help="Direct diagram generation query"),
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Diagram type (flowchart, network, etc.)"),
    theme: str = typer.Option("default", "--theme", help="Color theme (default, professional, vibrant)"),
    layout: str = typer.Option("dagre", "--layout", "-l", help="Layout engine (dagre, elk)"),
    from_search: Optional[str] = typer.Option(None, "--search", "-s", help="Create diagram from document search"),
    export_format: Optional[str] = typer.Option(None, "--export", "-e", help="Export format (png, pdf, etc.)")
):
    """Generate diagrams from natural language or documents"""
    print_header("Diagram Generator")
    diagram_generator = LazyLoader.get_diagram_generator()
    
    # Check if D2 is available
    if not diagram_generator.d2_available:
        console.print("[red]❌ D2 diagram tool not found![/red]")
        console.print("\nInstall D2 from: https://d2lang.com/tour/install")
        console.print("Example: curl -fsSL https://d2lang.com/install.sh | sh -s --")
        raise typer.Exit(1)
    
    # Validate layout
    if layout not in ["dagre", "elk"]:
        console.print(f"[yellow]Invalid layout '{layout}', using 'dagre'[/yellow]")
        layout = "dagre"
    
    # If no arguments, enter interactive mode
    if not query and not from_search:
        diagram_generator.interactive_diagram_generator()
        return
    
    # Direct generation mode
    if from_search:
        # Generate from document search
        result = diagram_generator.create_diagram_from_documents(
            from_search,
            diagram_type=type or "auto",
            theme=theme
        )
    elif query:
        # Generate from query
        result = diagram_generator.generate_diagram_from_query(
            query,
            diagram_type=type or "auto",
            use_context=True,
            layout=layout,
            theme=theme
        )
    else:
        console.print("[yellow]Please provide a query or use --search option[/yellow]")
        raise typer.Exit(1)
    
    # Handle result
    if result.success:
        console.print(f"\n[green]✅ Diagram generated successfully![/green]")
        console.print(f"📁 Saved to: [cyan]{result.output_path}[/cyan]")
        
        # Export if requested
        if export_format and result.output_path:
            exported = diagram_generator.export_diagram(
                result.output_path,
                export_format
            )
            if exported:
                console.print(f"📤 Exported to: [cyan]{exported}[/cyan]")
        
        # Show metadata
        if result.metadata:
            console.print(f"\nType: {result.metadata.get('diagram_type', 'unknown')}")
            console.print(f"Layout: {result.metadata.get('layout', 'unknown')}")
            console.print(f"Theme: {result.metadata.get('theme', 'default')}")
    else:
        console.print(f"\n[red]❌ Diagram generation failed: {result.error_message}[/red]")
        raise typer.Exit(1)

@app.command()
def diagram_batch(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="File containing diagram queries (one per line)"),
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Default diagram type for all"),
    theme: str = typer.Option("default", "--theme", help="Color theme")
):
    """Generate multiple diagrams in batch mode"""
    print_header("Batch Diagram Generation")
    diagram_generator = LazyLoader.get_diagram_generator()
    
    if not diagram_generator.d2_available:
        console.print("[red]❌ D2 diagram tool not found![/red]")
        raise typer.Exit(1)
    
    queries = []
    
    if file and file.exists():
        # Read queries from file
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse line for type hints
                    if ':' in line:
                        dtype, query = line.split(':', 1)
                        queries.append({
                            "query": query.strip(),
                            "diagram_type": dtype.strip(),
                            "theme": theme
                        })
                    else:
                        queries.append({
                            "query": line,
                            "diagram_type": type or "auto",
                            "theme": theme
                        })
    else:
        # Interactive input
        console.print("Enter diagram queries (one per line, empty line to finish):")
        console.print("[dim]Format: [type:]description (e.g., 'flowchart:login process')[/dim]\n")
        
        while True:
            line = console.input(f"{len(queries)+1}. ").strip()
            if not line:
                break
            
            if ':' in line:
                dtype, query = line.split(':', 1)
                queries.append({
                    "query": query.strip(),
                    "diagram_type": dtype.strip(),
                    "theme": theme
                })
            else:
                queries.append({
                    "query": line,
                    "diagram_type": type or "auto",
                    "theme": theme
                })
    
    if not queries:
        console.print("[yellow]No queries provided[/yellow]")
        return
    
    # Generate diagrams
    console.print(f"\n[bold]Generating {len(queries)} diagrams...[/bold]")
    results = diagram_generator.batch_generate_diagrams(queries)
    
    # Show results
    success_count = sum(1 for r in results if r.success)
    console.print(f"\n[bold]Batch Complete![/bold]")
    console.print(f"[green]✅ Success: {success_count}/{len(results)}[/green]")
    
    if success_count > 0:
        console.print("\n[bold]Generated Diagrams:[/bold]")
        for i, (query, result) in enumerate(zip(queries, results)):
            if result.success:
                console.print(f"  {i+1}. [green]✓[/green] {query['query'][:50]}... → [cyan]{result.output_path.name}[/cyan]")
    
    if success_count < len(results):
        console.print("\n[bold]Failed Diagrams:[/bold]")
        for i, (query, result) in enumerate(zip(queries, results)):
            if not result.success:
                console.print(f"  {i+1}. [red]✗[/red] {query['query'][:50]}... - {result.error_message}")

@app.command()
def diagram_gallery(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of diagrams to show"),
    format: Optional[str] = typer.Option(None, "--format", "-f", help="Filter by format (svg, png, pdf)"),
    sort: str = typer.Option("modified", "--sort", "-s", help="Sort by: modified, created, name, size")
):
    """Browse and manage generated diagrams"""
    print_header("Diagram Gallery")
    diagram_generator = LazyLoader.get_diagram_generator()
    
    # Get diagrams
    diagrams = diagram_generator.list_available_diagrams(
        sort_by=sort,
        filter_format=format
    )[:limit]
    
    if not diagrams:
        console.print("[yellow]No diagrams found[/yellow]")
        console.print("\nGenerate diagrams using:")
        console.print("  [cyan]research diagram \"your description here\"[/cyan]")
        return
    
    # Create table
    table = Table(title=f"Generated Diagrams (showing {len(diagrams)})", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Filename", style="cyan", no_wrap=True)
    table.add_column("Format", style="magenta", width=6)
    table.add_column("Size", style="green", justify="right")
    table.add_column("Modified", style="yellow")
    table.add_column("Info", style="blue")
    
    for i, diagram in enumerate(diagrams, 1):
        info = diagram_generator.get_diagram_info(diagram)
        if info:
            # Build info string
            info_parts = []
            if info.get("has_source"):
                info_parts.append("📄")
            if info.get("nodes"):
                info_parts.append(f"{info['nodes']}n")
            if info.get("connections"):
                info_parts.append(f"{info['connections']}c")
            
            table.add_row(
                str(i),
                info['filename'],
                info['format'].upper(),
                info['size_human'],
                info['modified'].strftime('%Y-%m-%d %H:%M'),
                " ".join(info_parts)
            )
    
    console.print(table)
    console.print("\n[dim]Legend: 📄=has source, n=nodes, c=connections[/dim]")
    console.print("[dim]Use 'research diagram' for interactive mode[/dim]")

@app.command()
def update(
    path: str = typer.Argument(..., help="Path to directory to update"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R", help="Recursively check subdirectories"),
    max_workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Number of threads (default: auto-detect optimal)"),
    disable_threading: bool = typer.Option(False, "--no-threading", help="Disable multi-threading (single-threaded mode)")
):
    """Update index with only new or modified files"""
    print_header("Index Update")
    parser = LazyLoader.get_parser()
    path_obj = Path(path).expanduser()
    
    # Handle threading configuration
    if disable_threading:
        workers = 1
        console.print("[dim]Running in single-threaded mode[/dim]")
    elif max_workers is not None:
        if max_workers < 1:
            console.print("[red]Error: --workers must be at least 1[/red]")
            raise typer.Exit(1)
        workers = max_workers
        console.print(f"[dim]Using {workers} worker threads[/dim]")
    else:
        workers = None  # Auto-detect
        optimal = min(16, (os.cpu_count() or 1) * 2)
        console.print(f"[dim]Auto-detected {optimal} optimal worker threads[/dim]")
    
    try:
        if not path_obj.is_dir():
            console.print(f"[red]Path must be a directory: {path}[/red]")
            raise typer.Exit(1)
            
        stats = parser.update_directory(
            path_obj, 
            recursive=recursive,
            max_workers=workers
        )
        
        # Summary output
        console.print(f"\n[bold]Update Complete![/bold]")
        console.print(f"[green]✓ Updated: {stats['indexed']} files[/green]")
        console.print(f"[red]✗ Failed: {stats['failed']} files[/red]")
        
    except KeyboardInterrupt:
        console.print(f"\n[yellow]⚠️  Update interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if os.getenv('DEBUG'):
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@app.command()
def watch(
    path: str = typer.Argument(..., help="Directory to watch for changes"),
    interval: int = typer.Option(60, "--interval", "-i", help="Check interval in seconds")
):
    """Watch directory and auto-index new/changed files"""
    print_header("Directory Watcher")
    parser = LazyLoader.get_parser()
    directory = Path(path).expanduser()
    if not directory.is_dir():
        console.print(f"[red]Not a directory: {path}[/red]")
        raise typer.Exit(1)
    parser.watch_directory(directory, interval=interval)

@app.command()
def status():
    """Show database statistics and indexed documents"""
    print_header("Database Status")
    parser = LazyLoader.get_parser()
    from .config import DB_PATH
    stats = parser.get_stats()
    if stats['total_chunks'] == 0:
        console.print("[yellow]No documents indexed yet[/yellow]")
        console.print("\nGet started by running:")
        console.print("  [cyan]research index ~/Documents/YourFolder[/cyan]")
        return
    
    # Check diagram availability
    diagram_generator = LazyLoader.get_diagram_generator()
    diagram_count = len(diagram_generator.list_available_diagrams())
    
    summary = f"""[bold]Overview[/bold]
- Total documents: {stats['total_documents']}
- Total chunks: {stats['total_chunks']:,}
- Total size: {stats['total_size_mb']:.1f} MB
- Generated diagrams: {diagram_count}
- Database location: {DB_PATH}"""
    
    # Add D2 status
    if diagram_generator.d2_available:
        summary += "\n- D2 renderer: ✅ Available"
    else:
        summary += "\n- D2 renderer: ❌ Not installed"
    
    console.print(Panel(summary, title="Database Summary", padding=(1, 2)))
    
    if stats['total_documents'] <= 20:
        table = Table(title="\nIndexed Documents", show_lines=True)
    else:
        table = Table(title=f"\nIndexed Documents (showing 20 of {stats['total_documents']})", show_lines=True)
    table.add_column("Document", style="cyan", no_wrap=False)
    table.add_column("Type", style="green", width=6)
    table.add_column("Chunks", justify="right", style="yellow")
    table.add_column("Size (MB)", justify="right", style="blue")
    table.add_column("Indexed", style="dim")
    sorted_docs = sorted(stats['documents'].items())[:20]
    for doc_name, doc_info in sorted_docs:
        indexed_date = doc_info['indexed_at'].split('T')[0] if 'T' in doc_info['indexed_at'] else doc_info['indexed_at']
        table.add_row(doc_name, doc_info['file_type'], str(doc_info['chunks']), f"{doc_info['size_mb']:.1f}", indexed_date)
    console.print(table)

@app.command()
def list(
    sort: str = typer.Option("name", "--sort", "-s", help="Sort by: name, size, date, chunks"),
    filter: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter by filename pattern"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of documents to show")
):
    """List indexed documents with sorting and filtering"""
    print_header("Document List")
    parser = LazyLoader.get_parser()
    docs = parser.list_documents(sort_by=sort, filter_pattern=filter)
    if not docs:
        console.print("[yellow]No documents found[/yellow]")
        return
    showing = min(len(docs), limit)
    if len(docs) > limit:
        table = Table(title=f"Documents (showing {showing} of {len(docs)})")
    else:
        table = Table(title=f"Documents ({len(docs)} total)")
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green", width=6)
    table.add_column("Chunks", justify="right", style="yellow")
    table.add_column("Size (MB)", justify="right", style="blue")
    for i, doc in enumerate(docs[:limit], 1):
        table.add_row(str(i), doc['name'], doc['type'], str(doc['chunks']), f"{doc['size_mb']:.1f}")
    console.print(table)

@app.command()
def analyze(path: str = typer.Argument(..., help="Directory to analyze")):
    """Analyze directory without indexing (preview what would be indexed)"""
    print_header("Directory Analysis")
    parser = LazyLoader.get_parser()
    directory = Path(path).expanduser()
    if not directory.exists():
        console.print(f"[red]Directory not found: {path}[/red]")
        raise typer.Exit(1)
    parser.analyze_directory(directory)

@app.command()
def clear(yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt")):
    """Clear all indexed documents from the database"""
    print_header("Clear Database")
    parser = LazyLoader.get_parser()
    stats = parser.get_stats()
    if stats['total_chunks'] == 0:
        console.print("[yellow]Database is already empty[/yellow]")
        return
    console.print(f"[yellow]This will delete:[/yellow]")
    console.print(f"  • {stats['total_documents']} documents")
    console.print(f"  • {stats['total_chunks']:,} chunks")
    console.print(f"  • {stats['total_size_mb']:.1f} MB of indexed content")
    if not yes:
        confirm = typer.confirm("\nAre you sure you want to clear all indexed documents?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            return
    if parser.clear_all():
        console.print("[green]✓ Database cleared successfully[/green]")
    else:
        console.print("[red]✗ Failed to clear database[/red]")
        raise typer.Exit(1)

@app.command()
def remove(filename: str = typer.Argument(..., help="Document filename to remove from index")):
    """Remove a specific document from the index"""
    print_header("Remove Document")
    parser = LazyLoader.get_parser()
    if parser.remove_document(filename):
        console.print(f"[green]✓ Successfully removed: {filename}[/green]")
    else:
        raise typer.Exit(1)

@app.command()
def export(output: Optional[str] = typer.Option(None, "--output", "-o", help="Output filename for chat export")):
    """Export current chat conversation to markdown file"""
    print_header("Export Conversation")
    chat_interface = LazyLoader.get_chat_interface()
    filename = chat_interface.export_conversation(output)
    if filename:
        console.print(f"[green]✓ Exported to: {filename}[/green]")

@app.command()
def supported():
    """Show supported file types"""
    print_header("Supported File Types")
    from .config import SUPPORTED_EXTENSIONS
    console.print("\n[bold]The following file types can be indexed:[/bold]\n")
    
    file_type_descriptions = {
        '.pdf': "Portable Document Format",
        '.txt': "Plain text files",
        '.md': "Markdown documents",
        '.py': "Python source code",
        '.js': "JavaScript source code",
        '.html': "HTML web pages",
        '.json': "JSON data files",
        '.yaml': "YAML configuration files",
        '.yml': "YAML configuration files",
        '.rst': "reStructuredText documents",
        '.rtf': "Rich Text Format",
        '.docx': "Microsoft Word documents",
        '.csv': "Comma-separated values"
    }
    
    for ext in sorted(SUPPORTED_EXTENSIONS):
        description = file_type_descriptions.get(ext, "Document file")
        console.print(f"  [green]•[/green] {ext} - {description}")
    
    console.print("\n[dim]More file types can be added in config.py[/dim]")

@app.command()
def info():
    """Show detailed system information and capabilities"""
    print_header("System Information")
    config = get_config()
    
    # Show config location
    console.print(f"\n[bold]Configuration[/bold]")
    console.print(f"Config file: [cyan]{config.config_file}[/cyan]")
    console.print(f"Config directory: [cyan]{config.config_dir}[/cyan]")
    
    # Validate config
    issues = config.validate_config()
    if issues:
        console.print(f"Config status: [red]Issues found ({len(issues)})[/red]")
    else:
        console.print(f"Config status: [green]✓ Valid[/green]")
    
    # Get components
    parser = LazyLoader.get_parser()
    diagram_generator = LazyLoader.get_diagram_generator()
    
    # System info
    info_sections = []
    
    # Database info
    try:
        stats = parser.get_stats()
        db_info = f"""[bold]Database[/bold]
• Documents: {stats['total_documents']}
• Chunks: {stats['total_chunks']:,}
• Size: {stats['total_size_mb']:.1f} MB"""
        info_sections.append(db_info)
    except:
        info_sections.append("[bold]Database[/bold]\n• Status: Not initialized")
    
    # AI Models
    ai_info = "[bold]AI Models[/bold]"
    try:
        from .config import GEMINI_MODEL
        ai_info += f"\n• Gemini: {GEMINI_MODEL} ✅"
    except:
        ai_info += "\n• Gemini: Not configured ❌"
    info_sections.append(ai_info)
    
    # Diagram Tools
    diagram_info = "[bold]Diagram Generation[/bold]"
    if diagram_generator.d2_available:
        diagram_count = len(diagram_generator.list_available_diagrams())
        diagram_info += f"\n• D2 Renderer: Available ✅"
        diagram_info += f"\n• Generated diagrams: {diagram_count}"
    else:
        diagram_info += "\n• D2 Renderer: Not installed ❌"
    info_sections.append(diagram_info)
    
    # System Resources
    import psutil
    cpu_count = os.cpu_count() or 1
    memory = psutil.virtual_memory()
    resource_info = f"""[bold]System Resources[/bold]
• CPU cores: {cpu_count}
• Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available
• Optimal threads: {min(32, cpu_count * 4)} for indexing"""
    info_sections.append(resource_info)
    
    # Display all sections
    for section in info_sections:
        console.print(Panel(section, padding=(1, 2)))
        console.print()

@app.callback()
def main_callback(
    version: bool = typer.Option(False, "--version", "-v", help="Show version information"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode")
):
    """Research document chat CLI - Index and chat with your documents using AI"""
    if debug:
        os.environ['DEBUG'] = '1'
        console.print("[dim]Debug mode enabled[/dim]")
    
    if version:
        console.print("[bold]Research CLI[/bold] version 1.1.0")
        console.print("Built with: ChromaDB, Gemini AI, Rich, D2")
        console.print("\n[bold]Features:[/bold]")
        console.print("• Document indexing and search")
        console.print("• AI-powered chat interface")
        console.print("• Diagram generation from text/documents")
        console.print("• Multi-threaded processing")
        console.print("• Web and GitHub indexing")
        raise typer.Exit()

create_config_commands(app)
# Entry point
if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        # Use print_exception for proper error display
        console.print("\n[red]Unexpected error occurred[/red]")
        console.print_exception(show_locals=os.getenv('DEBUG') is not None)
        sys.exit(1)