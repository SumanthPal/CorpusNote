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
from config import *

console = Console()

class ChatInterface:
    def __init__(self):
        """Initialize chat interface with Gemini and ChromaDB"""
        # Initialize Gemini
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in config.py")
        
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
        
        # Chat history for context
        self.history = []
        self.max_history = 10  # Keep last 10 exchanges
    
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
        
        try:
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.8,
                    max_output_tokens=2048,
                )
            )
            
            return response.text
            
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
        """Run interactive chat session with enhanced image support"""
        if not self.check_collection():
            return
        
        # Display welcome message with image support info
        console.print(Panel.fit(
            "[bold green]Research Assistant[/bold green]\n"
            "Chat with your indexed documents including images, diagrams, and text. "
            "I can help you find and understand information from all your research materials.\n\n"
            "Commands:\n"
            "• Type your question and press Enter\n"
            "• 'clear' - Clear conversation history\n"
            "• 'sources' - Toggle source display\n"
            "• 'stats' - Show database statistics\n"
            "• 'filter images' - Search only image content\n"
            "• 'filter text' - Search only text documents\n"
            "• 'clear filter' - Remove content type filters\n"
            "• 'export' - Export conversation to file\n"
            "• 'exit' or Ctrl+C - Exit chat",
            title="Welcome"
        ))
        
        # Show current status with content type breakdown
        self.show_collection_summary()
        
        if document_filter:
            console.print(f"[dim]Filter active: {document_filter}[/dim]")
        
        # Chat settings
        show_sources = True
        content_filter = None
        
        # Main chat loop
        while True:
            try:
                # Get user input
                console.print()
                query = Prompt.ask("[bold cyan]You[/bold cyan]")
                
                # Handle commands
                if query.lower() in ['exit', 'quit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                elif query.lower() == 'clear':
                    self.clear_history()
                    continue
                
                elif query.lower() == 'sources':
                    show_sources = not show_sources
                    status = "enabled" if show_sources else "disabled"
                    console.print(f"[yellow]Source display {status}[/yellow]")
                    continue
                
                elif query.lower() == 'stats':
                    self.show_stats()
                    continue
                
                elif query.lower() in ['filter images', 'filter image']:
                    content_filter = 'images'
                    console.print("[yellow]Now filtering to image content only[/yellow]")
                    continue
                
                elif query.lower() in ['filter text', 'filter documents']:
                    content_filter = 'text'
                    console.print("[yellow]Now filtering to text documents only[/yellow]")
                    continue
                
                elif query.lower() in ['clear filter', 'no filter']:
                    content_filter = None
                    console.print("[yellow]Content type filter cleared[/yellow]")
                    continue
                
                elif query.lower() == 'export':
                    self.export_conversation()
                    continue
                
                elif query.strip() == '':
                    continue
                
                # Process the query with current filters
                active_filter = content_filter or document_filter
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
    
    def show_collection_summary(self):
        """Show brief collection summary with content type breakdown"""
        if not self.check_collection():
            return
        
        all_data = self.collection.get()
        total_chunks = len(all_data['ids'])
        
        # Count content types
        content_types = {}
        documents = set()
        
        for metadata in all_data['metadatas']:
            content_type = metadata.get('content_type', 'text')
            content_types[content_type] = content_types.get(content_type, 0) + 1
            documents.add(metadata.get('source', 'Unknown'))
        
        # Display summary
        summary = f"Connected to database: {total_chunks} chunks from {len(documents)} documents"
        
        if content_types:
            type_info = []
            for ctype, count in content_types.items():
                if ctype == 'image':
                    type_info.append(f"{count} image chunks")
                else:
                    type_info.append(f"{count} text chunks")
            summary += f" ({', '.join(type_info)})"
        
        console.print(f"[dim]{summary}[/dim]")
    
    def show_stats(self):
        """Show database statistics with content type breakdown"""
        if not self.check_collection():
            return
        
        # Get all metadata
        all_data = self.collection.get()
        
        # Calculate stats
        total_chunks = len(all_data['ids'])
        documents = {}
        content_types = {'text': 0, 'image': 0}
        file_types = {}
        
        for metadata in all_data['metadatas']:
            source = metadata.get('source', 'Unknown')
            content_type = metadata.get('content_type', 'text')
            file_type = metadata.get('file_type', 'unknown')
            
            # Track content types
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # Track file types
            file_types[file_type] = file_types.get(file_type, 0) + 1
            
            # Track documents
            if source not in documents:
                documents[source] = {
                    'chunks': 0,
                    'content_type': content_type,
                    'file_type': file_type
                }
            documents[source]['chunks'] += 1
        
        # Display comprehensive stats
        stats_text = f"[bold]Database Statistics[/bold]\n\n"
        stats_text += f"Total chunks: {total_chunks}\n"
        stats_text += f"Total documents: {len(documents)}\n"
        stats_text += f"Average chunks per document: {total_chunks / len(documents):.1f}\n\n"
        
        # Content type breakdown
        stats_text += f"[bold]Content Types:[/bold]\n"
        for ctype, count in content_types.items():
            percentage = (count / total_chunks) * 100
            stats_text += f"  {ctype.title()}: {count} chunks ({percentage:.1f}%)\n"
        
        # File type breakdown
        if file_types:
            stats_text += f"\n[bold]File Types:[/bold]\n"
            for ftype, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
                stats_text += f"  {ftype}: {count} files\n"
        
        console.print(Panel(stats_text, title="Database Statistics"))
        
        # Show document list (limited)
        if len(documents) <= 15:
            console.print("\n[bold]Indexed documents:[/bold]")
            for doc, info in sorted(documents.items()):
                icon = "📷" if info['content_type'] == 'image' else "📄"
                console.print(f"  {icon} {doc} ({info['chunks']} chunks)")
        else:
            console.print(f"\n[dim]Showing first 15 of {len(documents)} documents[/dim]")
            for doc, info in list(sorted(documents.items()))[:15]:
                icon = "📷" if info['content_type'] == 'image' else "📄"
                console.print(f"  {icon} {doc} ({info['chunks']} chunks)")
    
    def ask_single(self, question: str, filter_pattern: str = None, show_sources: bool = True) -> None:
        """Single question mode - for CLI non-interactive use"""
        result = self.chat(question, filter_pattern, show_sources)
        
        # Display answer
        console.print(f"\n[bold green]Answer:[/bold green]")
        console.print(Markdown(result['response']))
        
        # Display sources
        if show_sources and result['sources']:
            console.print()
            console.print(self.format_sources_table(result['sources']))
    
    def export_conversation(self, filename: str = None) -> str:
        """Export conversation history to file"""
        if not self.history:
            console.print("[yellow]No conversation history to export[/yellow]")
            return None
        
        if not filename:
            filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # Format as markdown
        content = "# Research Chat Export\n\n"
        content += f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        content += "---\n\n"
        
        for entry in self.history:
            timestamp = entry.get('timestamp', 'Unknown time')
            role = "**You:**" if entry['role'] == 'user' else "**Assistant:**"
            content += f"{role} _{timestamp}_\n\n{entry['content']}\n\n---\n\n"
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        console.print(f"[green]✓ Conversation exported to {filename}[/green]")
        return filename


# Standalone test function
if __name__ == "__main__":
    # Test the chat interface
    try:
        chat = ChatInterface()
        
        # Test single query
        console.print("[bold]Testing single query...[/bold]")
        result = chat.chat("What documents do you have about networking?")
        console.print(f"Response: {result['response'][:200]}...")
        console.print(f"Sources found: {len(result['sources'])}")
        
        # Test image-specific query
        console.print("\n[bold]Testing image query...[/bold]")
        result = chat.chat("Show me any diagrams or images", document_filter="images")
        console.print(f"Image results: {result['response'][:200]}...")
        
    except Exception as e:
        console.print(f"[red]Test failed: {e}[/red]")