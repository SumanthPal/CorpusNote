# chat.py
import google.generativeai as genai
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
import chromadb
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path
from .config import *
import subprocess
import shutil

console = Console()

class ChatInterface:
    def __init__(self):
        """Initialize chat interface with Gemini and ChromaDB"""
        # Initialize Gemini
        try:
            from corpus.model_manager import ModelsManager
            self.models_manager = ModelsManager()
            self.model = self.models_manager.get_active_model()
            
            if not self.model:
                raise ValueError("No active AI model configured. Run 'corpus models setup' first.")
        
        except ImportError:
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found and models manager not available")
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel(GEMINI_MODEL)
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=DB_PATH)
        try:
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
            self.collection_exists = True
        except:
            self.collection_exists = False
            console.print("[yellow]Warning: No document collection found. Index some documents first.[/yellow]")
        
        #creating a directry to store generated diagrams
        
            
        self.history = []  # Store conversation history 
        self.max_history = MAX_MEMORY  # Keep last 10 exchanges
    
    def check_collection(self) -> bool:
        """Check if collection exists and has documents"""
        if not self.collection_exists:
            console.print("[red]No documents indexed. Run 'index' command first.[/red]")
            return False
        
        count = self.collection.count()
        if count == 0:
            console.print("[red]No documents in database. Index some documents first.[/red]")
            return False
        
        return True
    
    def search_documents(self, query: str, filter_dict: dict = None, n_results: int = None) -> dict:
        """Search for relevant document chunks using ChromaDB"""
        if not self.check_collection():
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        try:
            # Use configured MAX_RESULTS or override
            num_results = n_results or MAX_RESULTS
            
            # Perform semantic search
            results = self.collection.query(
                query_texts=[query],
                n_results=num_results,
                where=filter_dict if filter_dict else None
            )
            
            return results
            
        except Exception as e:
            console.print(f"[red]Search error: {e}[/red]")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def format_context(self, results: dict, show_metadata: bool = True) -> Tuple[str, List[Dict]]:
        """Format search results into context for LLM with enhanced image handling"""
        if not results['documents'] or not results['documents'][0]:
            return "", []
        
        context_parts = []
        sources = []
        
        # Track content types for better organization
        text_docs = []
        image_docs = []
        
        # Process each result
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0],
            results['distances'][0] if results.get('distances') else [0] * len(results['documents'][0])
        )):
            # Extract source info - handle both string and int types
            chunk_index = metadata.get('chunk_index', 0)
            total_chunks = metadata.get('total_chunks', 1)
            content_type = metadata.get('content_type', 'text')
            
            # Ensure they're integers
            if isinstance(chunk_index, str):
                chunk_index = int(chunk_index) if chunk_index.isdigit() else 0
            if isinstance(total_chunks, str):
                total_chunks = int(total_chunks) if total_chunks.isdigit() else 1
            
            source_info = {
                "document": metadata.get('source', 'Unknown'),
                "page": metadata.get('page', 'Unknown'),
                "chunk": chunk_index + 1,
                "total_chunks": total_chunks,
                "relevance": 1 - distance,
                "content_type": content_type,
                "file_type": metadata.get('file_type', 'unknown')
            }
            
            # Add image-specific metadata if available
            if content_type == 'image':
                source_info.update({
                    "width": metadata.get('width'),
                    "height": metadata.get('height'),
                    "format": metadata.get('format')
                })
            
            sources.append(source_info)
            
            # Format context based on content type
            if content_type == 'image':
                # Enhanced formatting for image content
                image_info = ""
                if source_info.get('width') and source_info.get('height'):
                    image_info = f" ({source_info['width']}x{source_info['height']}, {source_info.get('format', 'unknown format')})"
                
                context_part = f"""[IMAGE: {source_info['document']}{image_info} | Chunk {source_info['chunk']}/{source_info['total_chunks']}]
{doc}
"""
                image_docs.append(context_part)
            else:
                # Standard text document formatting
                context_part = f"""[DOCUMENT: {source_info['document']} | {source_info['page']} | Chunk {source_info['chunk']}/{source_info['total_chunks']}]
{doc}
"""
                text_docs.append(context_part)
        
        # Organize context: images first, then text documents
        organized_context = []
        if image_docs:
            organized_context.extend(image_docs)
        if text_docs:
            organized_context.extend(text_docs)
        
        # Join with clear separators
        full_context = "\n---\n".join(organized_context)
        
        return full_context, sources
    
    def generate_response(self, query: str, context: str, sources: List[Dict]) -> str:
        """Generate response using Gemini with enhanced image awareness"""
        # Analyze sources to understand content types
        has_images = any(source.get('content_type') == 'image' for source in sources)
        has_text = any(source.get('content_type') != 'image' for source in sources)
        
        # Build conversation history context
        history_context = ""
        if self.history:
            history_context = "\n\nPrevious conversation:\n"
            for h in self.history[-4:]:  # Last 2 exchanges
                history_context += f"{h['role']}: {h['content'][:200]}...\n"
        
        # Create enhanced prompt with image awareness
        image_instructions = ""
        if has_images:
            image_instructions = """
IMPORTANT: Some of the provided context comes from images (diagrams, charts, figures, etc.). 
- Image content includes both OCR-extracted text and AI-generated descriptions
- When referencing image content, mention that it comes from an image/diagram/figure
- Be specific about visual elements described in the context
- If discussing relationships shown in diagrams, explain them clearly
"""
        
        prompt = f"""You are a helpful research assistant. Your task is to answer questions based ONLY on the provided document excerpts.

Important instructions:
1. Base your answer solely on the provided context
2. If the answer isn't in the context, say "I couldn't find information about that in the indexed documents"
3. When referencing information, mention which document it comes from
4. Be specific and accurate
5. If multiple documents discuss the topic, synthesize the information
6. {'IMAGE CONTENT: ' + image_instructions if has_images else ''}
{history_context}

Context from documents:
{context}

User Question: {query}

Please provide a comprehensive answer based on the above context:"""
        # Generate response using the active model
        try:
            if hasattr(self.model, 'generate_content'):
                # Gemini model
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.8,
                        max_output_tokens=2048,
                    )
                )
                return response.text
            else:
                # Other models (OpenAI, Phi-3, etc.)
                return self.model.generate(prompt, temperature=0.7, top_p=0.8, max_tokens=2048)
                
        except Exception as e:
            console.print(f"[red]Generation error: {e}[/red]")
            return f"I encountered an error generating a response: {str(e)}"
                
    def format_sources_table(self, sources: List[Dict]) -> Table:
        """Create a pretty table of sources with enhanced image info"""
        table = Table(title="Sources Used", show_lines=True)
        table.add_column("Document", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta", width=8)
        table.add_column("Location", style="green")
        table.add_column("Details", style="blue")
        table.add_column("Relevance", justify="right", style="yellow")
        
        for source in sources:
            # Determine content type display
            content_type = source.get('content_type', 'text')
            file_type = source.get('file_type', '').upper().replace('.', '')
            
            if content_type == 'image':
                type_display = f"📷 {file_type}"
            elif content_type == 'webpage':
                type_display = f'🌐 WEB'
            elif content_type in ['github_file', 'github_code', 'github_repo_info']:
                type_display = f'🐙 GITHUB'
            else:
                type_display = f"📄 {file_type}"
            
            
            # Format location
            if content_type == 'image':
                location = f"Image (chunk {source['chunk']}/{source['total_chunks']})"
            else:
                location = f"{source['page']} (chunk {source['chunk']}/{source['total_chunks']})"
            
            # Format details
            details = ""
            if content_type == 'image' and source.get('width') and source.get('height'):
                details = f"{source['width']}×{source['height']}"
                if source.get('format'):
                    details += f" {source['format']}"
            elif content_type != 'image':
                details = "Text content"
            
            # Relevance bar
            relevance_bar = "▰" * int(source['relevance'] * 10) + "▱" * (10 - int(source['relevance'] * 10))
            
            table.add_row(
                source['document'],
                type_display,
                location,
                details,
                f"{source['relevance']:.2f} {relevance_bar}"
            )
        
        return table
    
    def chat(self, query: str, document_filter: str = None, show_sources: bool = True) -> Dict:
        """Main chat function - single query/response"""
        if not self.check_collection():
            return {
                "response": "No documents indexed yet. Please index some documents first.",
                "sources": [],
                "error": True
            }
        
        # Build filter if specified
        filter_dict = None
        if document_filter:
            # Handle different filter types
            if document_filter.lower() in ['image', 'images']:
                filter_dict = {"content_type": "image"}
                console.print(f"[dim]Filtering to image content only[/dim]")
            elif document_filter.lower() in ['text', 'documents']:
                filter_dict = {"content_type": {"$ne": "image"}}
                console.print(f"[dim]Filtering to text documents only[/dim]")
            else:
                filter_dict = {"source": {"$contains": document_filter}}
                console.print(f"[dim]Filtering to documents containing: {document_filter}[/dim]")
        
        # Search for relevant chunks
        console.print("[dim]Searching documents...[/dim]")
        results = self.search_documents(query, filter_dict)
        
        # Format context and extract sources
        context, sources = self.format_context(results)
        
        if not context:
            return {
                "response": "I couldn't find any relevant information in the indexed documents for your query.",
                "sources": [],
                "error": False
            }
        
        # Generate response
        console.print("[dim]Generating response...[/dim]")
        response = self.generate_response(query, context, sources)
        
        # Update history
        self.add_to_history("user", query)
        self.add_to_history("assistant", response)
        
        return {
            "response": response,
            "sources": sources,
            "error": False
        }
    
    def add_to_history(self, role: str, content: str):
        """Add exchange to conversation history"""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep history size manageable
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
        console.print("[yellow]Conversation history cleared[/yellow]")
    
    def interactive_chat(self, document_filter: str = None):
        """Run interactive chat session with enhanced commands"""
        if not self.check_collection():
            return
        
        # Display enhanced welcome message
        self._show_help()
        
        # Show current status
        self.show_collection_summary()
        
        if document_filter:
            console.print(f"[dim]Filter active: {document_filter}[/dim]")
        
        # Chat settings
        show_sources = True
        content_filter = None
        focused_document = None
        
        # Main chat loop
        while True:
            try:
                # Get user input
                console.print()
                query = Prompt.ask("[bold cyan]You[/bold cyan]")
                
                # Parse commands
                query_lower = query.lower()
                query_parts = query.split()
                
                # Handle exit commands
                if query_lower in ['exit', 'quit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                # Clear history
                elif query_lower == 'clear':
                    self.clear_history()
                    continue
                
                # Toggle sources
                elif query_lower in ['sources', 'sources on', 'sources off']:
                    if query_lower == 'sources on':
                        show_sources = True
                    elif query_lower == 'sources off':
                        show_sources = False
                    else:
                        show_sources = not show_sources
                    status = "enabled" if show_sources else "disabled"
                    console.print(f"[yellow]Source display {status}[/yellow]")
                    continue
                
                # Show stats
                elif query_lower == 'stats':
                    self.show_stats()
                    continue
                
                # Search command
                elif query_lower.startswith('search '):
                    search_query = query[7:].strip()
                    if not search_query:
                        console.print("[yellow]Please provide a search query[/yellow]")
                        continue
                    
                    self._handle_search_command(search_query, focused_document)
                    continue
                
                # Summarize command
                elif query_lower.startswith('summarize '):
                    filename = query[10:].strip()
                    if not filename:
                        console.print("[yellow]Please provide a filename[/yellow]")
                        continue
                    
                    self._handle_summarize_command(filename)
                    continue
                
                # Preview command
                elif query_lower.startswith('preview '):
                    filename = query[8:].strip()
                    if not filename:
                        console.print("[yellow]Please provide a filename[/yellow]")
                        continue
                    
                    self._handle_preview_command(filename)
                    continue
                
                # Similar documents command
                elif query_lower.startswith('similar '):
                    filename = query[8:].strip()
                    if not filename:
                        console.print("[yellow]Please provide a filename[/yellow]")
                        continue
                    
                    self._handle_similar_command(filename)
                    continue
                
                # Extract command
                elif query_lower.startswith('extract '):
                    parts = query[8:].strip().split(' ', 1)
                    if len(parts) < 2:
                        console.print("[yellow]Usage: extract <filename> <topic>[/yellow]")
                        continue
                    
                    filename, topic = parts
                    self._handle_extract_command(filename, topic)
                    continue
                
                # Compare command
                elif query_lower.startswith('compare '):
                    parts = query[8:].strip().split()
                    if len(parts) < 2:
                        console.print("[yellow]Usage: compare <file1> <file2>[/yellow]")
                        continue
                    
                    file1, file2 = parts[0], parts[1]
                    self._handle_compare_command(file1, file2)
                    continue
                
                # Topics command
                elif query_lower == 'topics':
                    self._handle_topics_command()
                    continue
                
                # List command
                elif query_lower.startswith('list'):
                    pattern = query[4:].strip() if len(query) > 4 else None
                    self._handle_list_command(pattern)
                    continue
                
                # Filter commands
                elif query_lower in ['filter images', 'filter image']:
                    content_filter = 'images'
                    console.print("[yellow]Now filtering to image content only[/yellow]")
                    continue
                
                elif query_lower in ['filter text', 'filter documents']:
                    content_filter = 'text'
                    console.print("[yellow]Now filtering to text documents only[/yellow]")
                    continue
                
                elif query_lower in ['clear filter', 'no filter']:
                    content_filter = None
                    console.print("[yellow]Content type filter cleared[/yellow]")
                    continue
                
                # Focus command
                elif query_lower.startswith('focus '):
                    filename = query[6:].strip()
                    if filename:
                        focused_document = filename
                        console.print(f"[yellow]Focused on: {filename}[/yellow]")
                        console.print("[dim]Use 'focus clear' to remove focus[/dim]")
                    continue
                
                elif query_lower == 'focus clear':
                    focused_document = None
                    console.print("[yellow]Document focus cleared[/yellow]")
                    continue
                
                # Export command
                elif query_lower == 'export':
                    self.export_conversation()
                    continue
                
                # Help command
                elif query_lower in ['help', '?']:
                    self._show_help()
                    continue
                
                elif query.strip() == '':
                    continue
                
                # Add this section in your interactive_chat() method with the other elif commands:

# Model management commands
                elif query_lower.startswith('model '):
                    model_command = query[6:].strip()
                    
                    if model_command == 'list':
                        console.print(self.models_manager.list_models())
                        console.print("\n[dim]Use 'model switch <name>' to change models[/dim]")
                        
                    elif model_command.startswith('switch '):
                        model_name = model_command[7:].strip()
                        if not model_name:
                            console.print("[yellow]Usage: model switch <model_name>[/yellow]")
                            console.print("Available models:")
                            for name in self.models_manager.models.keys():
                                console.print(f"  • {name}")
                        else:
                            if self.models_manager.set_active_model(model_name):
                                self.model = self.models_manager.get_active_model()
                                if self.model:
                                    console.print(f"[green]✓ Switched to {model_name}[/green]")
                                else:
                                    console.print(f"[yellow]⚠ Switched to {model_name} but model needs configuration[/yellow]")
                                    console.print("Run 'model refresh' or check API keys")
                            else:
                                console.print(f"[red]✗ Failed to switch to {model_name}[/red]")
                                
                    elif model_command == 'info':
                        if self.model:
                            info = self.model.get_info()
                            console.print(f"\n[bold]Active Model:[/bold] {info['name']}")
                            console.print(f"Type: {info['type']}")
                            console.print(f"Model ID: {info['model_id']}")
                            console.print(f"Location: {info['location']}")
                            console.print(f"Context Length: {info['context_length']:,} tokens")
                            console.print(f"Supports Streaming: {info['supports_streaming']}")
                        else:
                            console.print("[yellow]No active model set[/yellow]")
                            
                    elif model_command == 'refresh':
                        console.print("[cyan]Refreshing models...[/cyan]")
                        self.models_manager.refresh_models()
                        self.model = self.models_manager.get_active_model()
                        
                    else:
                        console.print("[yellow]Unknown model command. Available:[/yellow]")
                        console.print("  • model list - Show available models")
                        console.print("  • model switch <name> - Switch models")
                        console.print("  • model info - Show current model")
                        console.print("  • model refresh - Refresh models")
                    
                    continue            
                
                # Process as regular chat query
                active_filter = focused_document or content_filter or document_filter
                with console.status("[bold green]Thinking...[/bold green]"):
                    result = self.chat(query, active_filter, show_sources)
                
                # Display response
                console.print(f"\n[bold green]Assistant:[/bold green]")
                console.print(Panel(Markdown(result['response']), padding=(1, 2)))
                
                # Display sources if enabled
                if show_sources and result['sources']:
                    console.print()
                    console.print(self.format_sources_table(result['sources']))
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def _handle_search_command(self, search_query: str, focused_document: str = None):
        """Handle document search within chat"""
        console.print(f"\n[bold]Searching for: '{search_query}'[/bold]")
        
        # Build filter if focused
        filter_dict = None
        if focused_document:
            filter_dict = {"source": focused_document}
            console.print(f"[dim]Searching within: {focused_document}[/dim]")
        
        # Perform search
        results = self.search_documents(search_query, filter_dict, n_results=10)
        
        if not results['documents'] or not results['documents'][0]:
            console.print("[yellow]No results found[/yellow]")
            return
        
        # Display results
        console.print(f"\nFound {len(results['documents'][0])} results:\n")
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0][:5],  # Show top 5
            results['metadatas'][0][:5],
            results['distances'][0][:5]
        ), 1):
            relevance = 1 - distance
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 'Unknown')
            
            # Highlight search terms
            preview = doc[:200] + "..." if len(doc) > 200 else doc
            for term in search_query.split():
                preview = preview.replace(term, f"[bold yellow]{term}[/bold yellow]")
            
            console.print(f"{i}. [cyan]{source}[/cyan] - {page}")
            console.print(f"   Relevance: {'█' * int(relevance * 10)}{'░' * (10 - int(relevance * 10))} {relevance:.2f}")
            console.print(f"   [dim]{preview}[/dim]\n")

    def _handle_summarize_command(self, filename: str):
        """Handle document summarization within chat"""
        console.print(f"\n[bold]Summarizing: {filename}[/bold]")
        
        # Get document chunks
        results = self.collection.get(where={"source": filename})
        
        if not results['documents']:
            # Try fuzzy match
            suggestions = self._find_similar_filenames(filename)
            if suggestions:
                console.print(f"[yellow]Document '{filename}' not found. Did you mean:[/yellow]")
                for i, suggestion in enumerate(suggestions[:5], 1):
                    console.print(f"  {i}. {suggestion}")
                console.print("[dim]Try: summarize <exact filename>[/dim]")
            else:
                console.print(f"[red]Document '{filename}' not found[/red]")
            return
        
        # Generate summary
        with console.status("[bold green]Generating summary..."):
            full_text = "\n".join(results['documents'])
            
            prompt = f"""Provide a comprehensive summary of this document in 2-3 paragraphs.
    Include main topics, key findings, and important details.

    Document: {filename}
    Content: {full_text[:8000]}"""
            
            response_text = self._generate_with_model(prompt)
        
        # Display summary
        metadata = results['metadatas'][0] if results['metadatas'] else {}
        console.print(f"\n[bold]Summary[/bold]")
        console.print(f"[dim]Type: {metadata.get('file_type', 'unknown')} | Chunks: {len(results['documents'])}[/dim]\n")
        console.print(Panel(Markdown(response_text), padding=(1, 2)))

    def _handle_preview_command(self, filename: str):
        """Show preview of document content"""
        results = self.collection.get(where={"source": filename}, limit=2)
        
        if not results['documents']:
            console.print(f"[red]Document '{filename}' not found[/red]")
            return
        
        # Show metadata
        metadata = results['metadatas'][0]
        console.print(f"\n[bold]Document Preview: {filename}[/bold]")
        console.print(f"Type: {metadata.get('file_type', 'unknown')} | ")
        console.print(f"Total chunks: {metadata.get('total_chunks', 'unknown')} | ")
        console.print(f"Indexed: {metadata.get('indexed_at', 'unknown')}\n")
        
        # Show first chunk
        preview = results['documents'][0]
        if len(preview) > 1000:
            preview = preview[:1000] + "..."
        
        console.print(Panel(preview, title="Content Preview", padding=(1, 2)))
        
        if len(results['documents']) > 1:
            console.print(f"\n[dim]... and {metadata.get('total_chunks', 1) - 1} more chunks[/dim]")

    def _handle_similar_command(self, filename: str):
        """Find similar documents"""
        # Get source document
        results = self.collection.get(where={"source": filename}, limit=1)
        
        if not results['documents']:
            console.print(f"[red]Document '{filename}' not found[/red]")
            return
        
        console.print(f"\n[bold]Finding documents similar to: {filename}[/bold]")
        
        # Search for similar
        with console.status("[bold green]Analyzing similarity..."):
            similar_results = self.collection.query(
                query_texts=[results['documents'][0]],
                n_results=15
            )
        
        # Display results
        seen = set()
        count = 0
        
        for metadata, distance in zip(similar_results['metadatas'][0], similar_results['distances'][0]):
            doc_name = metadata.get('source', '')
            if doc_name != filename and doc_name not in seen:
                seen.add(doc_name)
                similarity = 1 - distance
                
                count += 1
                console.print(f"\n{count}. [cyan]{doc_name}[/cyan]")
                console.print(f"   Similarity: {'█' * int(similarity * 10)}{'░' * (10 - int(similarity * 10))} {similarity:.2f}")
                console.print(f"   Type: {metadata.get('file_type', 'unknown')}")
                
                if count >= 5:
                    break

    def _handle_extract_command(self, filename: str, topic: str):
        """Extract specific information from document"""
        results = self.collection.get(where={"source": filename})
        
        if not results['documents']:
            console.print(f"[red]Document '{filename}' not found[/red]")
            return
        
        console.print(f"\n[bold]Extracting: '{topic}' from {filename}[/bold]")
        
        with console.status("[bold green]Extracting information..."):
            full_text = "\n".join(results['documents'][:10])
            
            prompt = f"""Extract all information about '{topic}' from this document.
    Organize the information clearly with sections if appropriate.
    If the topic is not found, say so clearly.

    Document: {filename}
    Content: {full_text}

    Extract information about: {topic}"""
            
            response_text = self._generate_with_model(prompt)
        
        console.print(Panel(Markdown(response_text), title=f"Extracted: {topic}", padding=(1, 2)))

    def _handle_compare_command(self, file1: str, file2: str):
        """Compare two documents"""
        # Get both documents
        doc1 = self.collection.get(where={"source": file1})
        doc2 = self.collection.get(where={"source": file2})
        
        if not doc1['documents']:
            console.print(f"[red]Document '{file1}' not found[/red]")
            return
        if not doc2['documents']:
            console.print(f"[red]Document '{file2}' not found[/red]")
            return
        
        console.print(f"\n[bold]Comparing: {file1} vs {file2}[/bold]")
        
        with console.status("[bold green]Analyzing documents..."):
            doc1_text = "\n".join(doc1['documents'][:5])
            doc2_text = "\n".join(doc2['documents'][:5])
            
            prompt = f"""Compare these two documents comprehensively.
    Structure your comparison as:
    1. Main similarities
    2. Key differences  
    3. Unique content in each
    4. Overall assessment

    Document 1: {file1}
    {doc1_text[:3000]}

    Document 2: {file2}
    {doc2_text[:3000]}"""
            
            response_text = self._generate_with_model(prompt)
        
        console.print(Panel(Markdown(response_text), title="Comparison Results", padding=(1, 2)))

    def _handle_topics_command(self):
        """Analyze topics across all documents"""
        console.print("\n[bold]Analyzing topics in database...[/bold]")
        
        # Sample documents
        with console.status("[bold green]Analyzing content..."):
            results = self.collection.get(limit=50)  # Sample
            
            if not results['documents']:
                console.print("[yellow]No documents to analyze[/yellow]")
                return
            
            sample_text = "\n".join(results['documents'][:20])
            
            prompt = f"""Analyze this content and identify the 10 main topics or themes.
    For each topic provide:
    - Topic name
    - Brief description
    - Key terms associated with it
    - Estimated prevalence

    Content sample from {len(results['documents'])} documents:
    {sample_text[:5000]}"""
            
            response_text = self.model.generate_content(prompt)
        
        console.print(Panel(Markdown(response_text), title="Topic Analysis", padding=(1, 2)))

    def _handle_list_command(self, pattern: str = None):
        """List documents with optional pattern filter"""
        all_data = self.collection.get()
        
        if not all_data['ids']:
            console.print("[yellow]No documents indexed[/yellow]")
            return
        
        # Get unique documents
        documents = {}
        for metadata in all_data['metadatas']:
            source = metadata.get('source', 'Unknown')
            
            # Apply pattern filter if provided
            if pattern and pattern.lower() not in source.lower():
                continue
            
            if source not in documents:
                documents[source] = {
                    'chunks': 0,
                    'type': metadata.get('file_type', 'unknown'),
                    'size': metadata.get('file_size', 0)
                }
            documents[source]['chunks'] += 1
        
        if not documents:
            console.print(f"[yellow]No documents matching '{pattern}'[/yellow]")
            return
        
        # Display as table
        console.print(f"\n[bold]Documents{f' matching {pattern}' if pattern else ''}:[/bold]")
        
        for i, (name, info) in enumerate(sorted(documents.items())[:20], 1):
            size_mb = info['size'] / (1024 * 1024) if isinstance(info['size'], (int, float)) else 0
            console.print(f"{i:2d}. [cyan]{name}[/cyan] ({info['type']}) - {info['chunks']} chunks, {size_mb:.1f}MB")

    def _find_similar_filenames(self, partial: str) -> List[str]:
        """Find filenames similar to the partial input"""
        all_data = self.collection.get()
        filenames = set()
        
        partial_lower = partial.lower()
        
        for metadata in all_data['metadatas']:
            source = metadata.get('source', '')
            if partial_lower in source.lower():
                filenames.add(source)
        
        return sorted(filenames)

    def _show_help(self):
        """Show detailed help for all commands"""
        help_text = """[bold]Available Commands:[/bold]
    [bold cyan]Chat Commands:[/bold cyan]
    - Just type your question - Ask anything about your documents
    - clear - Clear conversation history
    - sources on/off - Toggle source citations
    - stats - Show database statistics
    - export - Save conversation to file
    - help or ? - Show this help
    - exit - Exit chat

    [bold cyan]Search & Analysis:[/bold cyan]
    - search <query> - Search all documents for specific content
    - summarize <filename> - Generate AI summary of a document
    - preview <filename> - Show preview of document content
    - topics - Analyze main topics across all documents
    - extract <filename> <topic> - Extract specific information

    [bold cyan]Document Management:[/bold cyan]
    - list [pattern] - List indexed documents (with optional filter)
    - similar <filename> - Find documents similar to given one
    - compare <file1> <file2> - Compare two documents
    - focus <filename> - Focus chat on specific document
    - focus clear - Remove document focus

    [bold cyan]Filters:[/bold cyan]
    - filter images - Show only image content
    - filter text - Show only text documents
    - clear filter - Remove all filters

    
    [bold cyan]AI Model Commands:[/bold cyan]
    - model list - Show all available AI models and their status
    - model switch <name> - Switch to a different AI model
    - model info - Show current active model details
    - model refresh - Refresh models (pick up new API keys)


    [bold cyan]Tips:[/bold cyan]
    - Use quotes for multi-word searches: search "machine learning"
    - Partial filenames work: summarize report (matches report.pdf)
    - Commands are case-insensitive
    - Tab completion available for filenames (if enabled)"""
        
        console.print(Panel(help_text, title="Chat Help", padding=(1, 2)))
    
    def _generate_with_model(self, prompt: str, **kwargs) -> str:
        """Generate response with any model type"""
        try:
            if hasattr(self.model, 'generate_content'):
                # Gemini model
                response = self.model.generate_content(prompt)
                return response.text
            elif hasattr(self.model, 'generate'):
                # Other models (OpenAI, Phi-3, etc.)
                return self.model.generate(prompt, **kwargs)
            else:
                return "Error: Unknown model type"
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
    def show_collection_summary(self):
        """Display a summary of the document collection"""
        try:
            # Get collection stats
            count = self.collection.count()
            
            if count == 0:
                console.print("[yellow]No documents indexed yet.[/yellow]")
                console.print("\nGet started by running:")
                console.print("  [cyan]corpus index ~/Documents/YourFolder[/cyan]")
                return
            
            # Get all documents to show summary
            all_data = self.collection.get()
            
            # Count unique documents
            unique_docs = set()
            doc_types = {}
            total_size = 0
            
            for metadata in all_data['metadatas']:
                source = metadata.get('source', 'Unknown')
                unique_docs.add(source)
                
                # Count by file type
                file_type = metadata.get('file_type', 'unknown')
                doc_types[file_type] = doc_types.get(file_type, 0) + 1
                
                # Sum up size (only count once per document)
                if source not in unique_docs or len(unique_docs) == 1:
                    file_size = metadata.get('file_size', 0)
                    if isinstance(file_size, str):
                        try:
                            file_size = int(file_size)
                        except:
                            file_size = 0
                    total_size += file_size
            
            # Create summary panel
            summary_text = f"""[bold cyan]Document Collection Summary[/bold cyan]
            
    📚 Total Documents: {len(unique_docs)}
    📄 Total Chunks: {count}
    💾 Total Size: {total_size / (1024*1024):.1f} MB

    [bold]Document Types:[/bold]"""
            
            for doc_type, type_count in sorted(doc_types.items()):
                summary_text += f"\n  • {doc_type}: {type_count} chunks"
            
            console.print(Panel(summary_text, title="Collection Overview", padding=(1, 2)))
            
            # Show recent documents if collection is small
            if len(unique_docs) <= 10:
                console.print("\n[bold]Indexed Documents:[/bold]")
                for doc in sorted(unique_docs)[:10]:
                    console.print(f"  • {doc}")
            else:
                console.print(f"\n[dim]Showing 10 most recent of {len(unique_docs)} documents:[/dim]")
                # Get most recent documents
                recent_docs = []
                for metadata in all_data['metadatas']:
                    source = metadata.get('source', 'Unknown')
                    indexed_at = metadata.get('indexed_at', '')
                    if source not in [d[0] for d in recent_docs]:
                        recent_docs.append((source, indexed_at))
                
                # Sort by date and show top 10
                recent_docs.sort(key=lambda x: x[1], reverse=True)
                for doc, _ in recent_docs[:10]:
                    console.print(f"  • {doc}")
            
        except Exception as e:
            console.print(f"[red]Error getting collection summary: {e}[/red]")