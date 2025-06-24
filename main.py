#!/usr/bin/env python3
# main.py
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from typing import Optional
import sys
from web_parser import WebParser
import chromadb

from file_parser import FileParser
from chat import ChatInterface
from config import SUPPORTED_EXTENSIONS, DB_PATH, COLLECTION_NAME
from diagram_generator import DiagramGenerator


# Initialize CLI app
app = typer.Typer(
    name="research",
    help="Research document chat CLI - Index and chat with your documents using AI",
    add_completion=False
)

# Initialize components
console = Console()
parser = FileParser()
chat_interface = ChatInterface()
web_parser = WebParser(parser)

# Initialize ChromaDB client
try:
    db_client = chromadb.PersistentClient(path=str(DB_PATH))
except Exception as e:
    console.print(f"[red]Error initializing ChromaDB: {e}[/red]")
    db_client = None

# Initialize DiagramGenerator
d2_available = True  # You can add a check for D2 availability if needed
diagram_generator = DiagramGenerator(
    diagrams_path="diagrams",
    d2_available=d2_available,
    console=console,
    db_client=db_client,
    collection_name=COLLECTION_NAME
)


# Helper function for pretty headers
def print_header(text: str):
    console.print(Panel.fit(f"[bold cyan]{text}[/bold cyan]", padding=(0, 1)))

@app.command()
def index(
    path: str = typer.Argument(..., help="Path to file or directory to index"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R", help="Recursively index subdirectories"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-indexing of existing files"),
    pattern: Optional[str] = typer.Option(None, "--pattern", "-p", help="File pattern to match (e.g., '*atomic*.pdf')")
):
    """Index documents into the research database"""
    print_header("Document Indexing")

    path_obj = Path(path).expanduser()

    try:
        if path_obj.is_file():
            # Index single file
            success, message = parser.index_file(path_obj, force=force)
            if success:
                console.print(f"[green]✓ {message}[/green]")
            else:
                console.print(f"[red]✗ {message}[/red]")
                raise typer.Exit(1)

        elif path_obj.is_dir():
            # Index directory
            stats = parser.index_directory(path_obj, recursive=recursive, force=force, pattern=pattern)

            # Display results
            console.print(f"\n[bold]Indexing Complete![/bold]")
            console.print(f"[green]✓ Indexed: {stats['indexed']} files[/green]")
            console.print(f"[yellow]⟳ Skipped: {stats['skipped']} files[/yellow]")
            console.print(f"[red]✗ Failed: {stats['failed']} files[/red]")

        else:
            console.print(f"[red]Path not found: {path}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
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

    if 'github.com' in url:
        # GitHub repository
        console.print(f"[cyan]Detected GitHub repository[/cyan]")
        stats = web_parser.index_github_repo(url, github_token)

        console.print(f"\n[bold]GitHub Indexing Complete![/bold]")
        console.print(f"[green]✓ Indexed: {stats['indexed']} files[/green]")
        console.print(f"[red]✗ Failed: {stats['failed']} files[/red]")

    elif url.startswith(('http://', 'https://')):
        # Regular website
        if max_pages == 1:
            # Single page
            success = web_parser.index_webpage(url)
            if success:
                console.print(f"[green]✓ Successfully indexed {url}[/green]")
            else:
                console.print(f"[red]✗ Failed to index {url}[/red]")
                raise typer.Exit(1)
        else:
            # Crawl website
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

    # Handle different input formats
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
    chat_interface.interactive_chat(document_filter=filter)

@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about your documents"),
    filter: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter documents by filename pattern"),
    no_sources: bool = typer.Option(False, "--no-sources", help="Hide source citations")
):
    """Ask a single question about your documents (non-interactive)"""
    chat_interface.ask_single(question, filter_pattern=filter, show_sources=not no_sources)

@app.command()
def diagram():
    """Enter interactive diagram generation mode"""
    print_header("Diagram Generator")
    diagram_generator.interactive_diagram_generator()

@app.command()
def update(
    path: str = typer.Argument(..., help="Directory to update"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R", help="Recursively check subdirectories")
):
    """Update index with new or modified files only"""
    print_header("Updating Index")

    directory = Path(path).expanduser()
    if not directory.is_dir():
        console.print(f"[red]Not a directory: {path}[/red]")
        raise typer.Exit(1)

    stats = parser.update_directory(directory, recursive=recursive)

    # Display results
    console.print(f"\n[bold]Update Complete![/bold]")
    console.print(f"[green]✓ Indexed: {stats['indexed']} new/modified files[/green]")
    console.print(f"[yellow]⟳ Skipped: {stats['skipped']} files[/yellow]")
    console.print(f"[red]✗ Failed: {stats['failed']} files[/red]")

@app.command()
def watch(
    path: str = typer.Argument(..., help="Directory to watch for changes"),
    interval: int = typer.Option(60, "--interval", "-i", help="Check interval in seconds")
):
    """Watch directory and auto-index new/changed files"""
    print_header("Directory Watcher")

    directory = Path(path).expanduser()
    if not directory.is_dir():
        console.print(f"[red]Not a directory: {path}[/red]")
        raise typer.Exit(1)

    parser.watch_directory(directory, interval=interval)

@app.command()
def status():
    """Show database statistics and indexed documents"""
    print_header("Database Status")

    stats = parser.get_stats()

    if stats['total_chunks'] == 0:
        console.print("[yellow]No documents indexed yet[/yellow]")
        console.print("\nGet started by running:")
        console.print("  [cyan]research index ~/Documents/YourFolder[/cyan]")
        return

    # Summary panel
    summary = f"""[bold]Overview[/bold]
- Total documents: {stats['total_documents']}
- Total chunks: {stats['total_chunks']:,}
- Total size: {stats['total_size_mb']:.1f} MB
- Database location: {DB_PATH}"""

    console.print(Panel(summary, title="Database Summary", padding=(1, 2)))

    # Document table
    if stats['total_documents'] <= 20:
        table = Table(title="\nIndexed Documents", show_lines=True)
    else:
        table = Table(title=f"\nIndexed Documents (showing 20 of {stats['total_documents']})", show_lines=True)

    table.add_column("Document", style="cyan", no_wrap=False)
    table.add_column("Type", style="green", width=6)
    table.add_column("Chunks", justify="right", style="yellow")
    table.add_column("Size (MB)", justify="right", style="blue")
    table.add_column("Indexed", style="dim")

    # Sort by name and show top 20
    sorted_docs = sorted(stats['documents'].items())[:20]

    for doc_name, doc_info in sorted_docs:
        indexed_date = doc_info['indexed_at'].split('T')[0] if 'T' in doc_info['indexed_at'] else doc_info['indexed_at']
        table.add_row(
            doc_name,
            doc_info['file_type'],
            str(doc_info['chunks']),
            f"{doc_info['size_mb']:.1f}",
            indexed_date
        )

    console.print(table)

@app.command()
def list(
    sort: str = typer.Option("name", "--sort", "-s", help="Sort by: name, size, date, chunks"),
    filter: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter by filename pattern"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of documents to show")
):
    """List indexed documents with sorting and filtering"""
    print_header("Document List")

    docs = parser.list_documents(sort_by=sort, filter_pattern=filter)

    if not docs:
        console.print("[yellow]No documents found[/yellow]")
        return

    # Create table
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
        table.add_row(
            str(i),
            doc['name'],
            doc['type'],
            str(doc['chunks']),
            f"{doc['size_mb']:.1f}"
        )

    console.print(table)

@app.command()
def analyze(
    path: str = typer.Argument(..., help="Directory to analyze"),
):
    """Analyze directory without indexing (preview what would be indexed)"""
    print_header("Directory Analysis")

    directory = Path(path).expanduser()
    if not directory.exists():
        console.print(f"[red]Directory not found: {path}[/red]")
        raise typer.Exit(1)

    parser.analyze_directory(directory)

@app.command()
def clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt")
):
    """Clear all indexed documents from the database"""
    print_header("Clear Database")

    stats = parser.get_stats()

    if stats['total_chunks'] == 0:
        console.print("[yellow]Database is already empty[/yellow]")
        return

    # Show what will be deleted
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
def remove(
    filename: str = typer.Argument(..., help="Document filename to remove from index")
):
    """Remove a specific document from the index"""
    print_header("Remove Document")

    if parser.remove_document(filename):
        console.print(f"[green]✓ Successfully removed: {filename}[/green]")
    else:
        raise typer.Exit(1)

@app.command()
def export(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output filename for chat export")
):
    """Export current chat conversation to markdown file"""
    print_header("Export Conversation")

    filename = chat_interface.export_conversation(output)
    if filename:
        console.print(f"[green]✓ Exported to: {filename}[/green]")

@app.command()
def supported():
    """Show supported file types"""
    print_header("Supported File Types")

    console.print("\n[bold]The following file types can be indexed:[/bold]\n")

    for ext in sorted(SUPPORTED_EXTENSIONS):
        if ext == '.pdf':
            console.print(f"  [green]•[/green] {ext} - Portable Document Format")
        elif ext == '.txt':
            console.print(f"  [green]•[/green] {ext} - Plain text files")
        elif ext == '.md':
            console.print(f"  [green]•[/green] {ext} - Markdown documents")
        else:
            console.print(f"  [green]•[/green] {ext}")

    console.print("\n[dim]More file types can be added in config.py[/dim]")

@app.callback()
def main_callback(
    version: bool = typer.Option(False, "--version", "-v", help="Show version information")
):
    """Research document chat CLI - Index and chat with your documents using AI"""
    if version:
        console.print("[bold]CORPUS CLI[/bold] version 1.0.0")
        console.print("Built with: ChromaDB, Gemini AI, Rich")
        raise typer.Exit()

# Entry point
if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        sys.exit(1)
