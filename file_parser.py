# file_parser.py
import chromadb
from pathlib import Path
from pypdf import PdfReader
from rich.console import Console
from rich.progress import track, Progress, SpinnerColumn, TextColumn
from rich.table import Table
import hashlib
from datetime import datetime
import re
from typing import List, Dict, Optional, Tuple
from config import *

console = Console()

class FileParser:
    def __init__(self):
        """Initialize the file parser with ChromaDB"""
        console.print("[cyan]Initializing ChromaDB...[/cyan]")
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Research document collection"}
        )
        console.print(f"[green]✓ Connected to database at {DB_PATH}[/green]")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate unique hash for file (for deduplication)"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _check_file_size(self, file_path: Path) -> bool:
        """Check if file size is within limits"""
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            console.print(f"[yellow]Skipping {file_path.name}: File too large ({size_mb:.1f}MB)[/yellow]")
            return False
        return True
    
    def _is_file_indexed(self, file_hash: str) -> bool:
        """Check if file is already indexed"""
        existing = self.collection.get(
            where={"file_hash": file_hash},
            limit=1
        )
        return len(existing['ids']) > 0
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text to handle common PDF issues"""
        # Remove null bytes and other problematic characters
        text = text.replace('\x00', '')
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce multiple newlines
        
        # Remove extremely long "words" (likely extraction errors)
        words = text.split()
        cleaned_words = [w for w in words if len(w) < 50]
        
        if len(cleaned_words) < len(words) * 0.5:
            # If we removed too many words, keep original
            return text
        
        return ' '.join(cleaned_words)
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text based on file type"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self._extract_pdf_text(file_path)
        elif suffix in ['.txt', '.md']:
            return self._extract_plain_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF with multiple fallback methods"""
        try:
            reader = PdfReader(file_path)
            num_pages = len(reader.pages)
            console.print(f"[dim]Processing {num_pages} pages from {file_path.name}[/dim]")
            
            text_parts = []
            extracted_pages = 0
            
            for i, page in enumerate(reader.pages):
                try:
                    # Try normal extraction
                    page_text = page.extract_text()
                    
                    # Clean the extracted text
                    if page_text:
                        page_text = self._clean_text(page_text)
                        
                        # Check if we got meaningful content
                        if len(page_text.strip()) > 20:  # At least 20 chars
                            text_parts.append(f"\n[Page {i+1}]\n{page_text}")
                            extracted_pages += 1
                        else:
                            console.print(f"[yellow]Page {i+1}: Minimal text extracted[/yellow]")
                    
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not extract page {i+1}: {e}[/yellow]")
            
            console.print(f"[dim]Successfully extracted text from {extracted_pages}/{num_pages} pages[/dim]")
            
            full_text = "\n".join(text_parts)
            
            # Final check
            if len(full_text.strip()) < 100:
                console.print(f"[red]Warning: Very little text extracted from {file_path.name}[/red]")
                console.print(f"[dim]Extracted text length: {len(full_text)} characters[/dim]")
            
            return full_text
            
        except Exception as e:
            console.print(f"[red]Error reading PDF {file_path.name}: {e}[/red]")
            return ""
    
    def _extract_plain_text(self, file_path: Path) -> str:
        """Extract text from plain text files"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    text = file_path.read_text(encoding=encoding)
                    return self._clean_text(text)
                except UnicodeDecodeError:
                    continue
            
            console.print(f"[yellow]Warning: Could not decode {file_path.name}[/yellow]")
            return ""
            
        except Exception as e:
            console.print(f"[red]Error reading text file {file_path.name}: {e}[/red]")
            return ""
    
    def _chunk_text(self, text: str, file_name: str) -> List[Dict[str, any]]:
        """
        Smart chunking that preserves context and handles edge cases
        Returns list of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        # Check total text length
        console.print(f"[dim]Text length for {file_name}: {len(text)} characters[/dim]")
        
        # Split into sentences for better chunking
        # Simple sentence splitter (you can make this more sophisticated)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        current_page = "Unknown"
        
        for sentence in sentences:
            # Check for page markers
            page_match = re.match(r'\[Page (\d+)\]', sentence.strip())
            if page_match:
                current_page = f"Page {page_match.group(1)}"
                continue
            
            # Skip very short sentences
            if len(sentence.strip()) < 10:
                continue
            
            words = sentence.split()
            word_count = len(words)
            
            # If this sentence would make chunk too large, save current chunk
            if current_size + word_count > CHUNK_SIZE and current_chunk:
                chunk_text = ' '.join(current_chunk)
                
                # Only add chunks that meet minimum length
                if len(chunk_text) >= MIN_CHUNK_LENGTH:
                    chunks.append({
                        'text': chunk_text.strip(),
                        'page': current_page,
                        'chunk_index': len(chunks)
                    })
                
                # Start new chunk with overlap
                if CHUNK_OVERLAP > 0 and len(current_chunk) > CHUNK_OVERLAP:
                    # Take last few sentences for overlap
                    overlap_start = max(0, len(current_chunk) - CHUNK_OVERLAP)
                    current_chunk = current_chunk[overlap_start:]
                    current_size = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += word_count
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= MIN_CHUNK_LENGTH:
                chunks.append({
                    'text': chunk_text.strip(),
                    'page': current_page,
                    'chunk_index': len(chunks)
                })
        
        console.print(f"[green]Created {len(chunks)} chunks from {file_name}[/green]")
        
        # Debug: Show first chunk if available
        if chunks:
            console.print(f"[dim]First chunk preview: {chunks[0]['text'][:100]}...[/dim]")
        
        return chunks
    
    def _safe_metadata(self, metadata: dict) -> dict:
        """Ensure all metadata values are safe for ChromaDB"""
        safe_meta = {}
        for key, value in metadata.items():
            # Convert None to empty string
            if value is None:
                safe_meta[key] = ""
            # Keep booleans as booleans
            elif isinstance(value, bool):
                safe_meta[key] = value
            # Convert numbers to their native type (int/float)
            elif isinstance(value, (int, float)):
                safe_meta[key] = value
            # Everything else to string
            else:
                safe_meta[key] = str(value)
        return safe_meta



    
    def index_file(self, file_path: Path, force: bool = False) -> Tuple[bool, str]:
        """
        Index a single file
        Returns: (success, message)
        """
        try:
            # Validate file
            if not file_path.exists():
                return False, f"File not found: {file_path}"
            
            if not file_path.is_file():
                return False, f"Not a file: {file_path}"
            
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                return False, f"Unsupported file type: {file_path.suffix}"
            
            # Check file size
            if not self._check_file_size(file_path):
                return False, f"File too large: {file_path.name}"
            
            # Check if already indexed
            file_hash = self._get_file_hash(file_path)
            if not force and self._is_file_indexed(file_hash):
                return False, f"Already indexed: {file_path.name}"
            
            # Extract text
            console.print(f"[cyan]Extracting text from {file_path.name}...[/cyan]")
            text = self._extract_text(file_path)
            
            if not text or len(text.strip()) < 50:
                return False, f"Insufficient text extracted from: {file_path.name} (only {len(text)} chars)"
            
            # Chunk text
            chunks = self._chunk_text(text, file_path.name)
            if not chunks:
                return False, f"No valid chunks created from: {file_path.name}"
            
            # Prepare for ChromaDB
            ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
            documents = [chunk['text'] for chunk in chunks]
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": file_path.name,
                    "full_path": str(file_path.absolute()),
                    "file_hash": file_hash,
                    "chunk_index": i,  # Keep as int
                    "total_chunks": len(chunks),  # Keep as int
                    "page": chunk.get('page', 'Unknown'),
                    "file_size": file_path.stat().st_size,  # Keep as int
                    "file_type": file_path.suffix.lower(),
                    "indexed_at": datetime.now().isoformat()
                }
                # Ensure all metadata is safe
                metadatas.append(self._safe_metadata(metadata))

            # Add to collection
            if force:
                # Remove existing chunks for this file
                self.collection.delete(where={"file_hash": file_hash})
            
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            
            return True, f"Indexed {file_path.name} ({len(chunks)} chunks)"
            
        except Exception as e:
            console.print(f"[red]Exception details: {type(e).__name__}: {str(e)}[/red]")
            return False, f"Failed to index {file_path.name}: {str(e)}"
    
    def get_stats(self) -> Dict:
        """Get detailed database statistics"""
        all_data = self.collection.get()
        
        if not all_data['ids']:
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "total_size_mb": 0,
                "documents": {}
            }
        
        documents = {}
        
        for metadata in all_data['metadatas']:
            source = metadata.get('source', 'Unknown')
            
            # Handle file_size as either int or string
            file_size = metadata.get('file_size', 0)
            if isinstance(file_size, str):
                try:
                    file_size = int(file_size)
                except ValueError:
                    file_size = 0
            
            if source not in documents:
                documents[source] = {
                    "chunks": 0,
                    "file_type": metadata.get('file_type', 'unknown'),
                    "size_mb": file_size / (1024 * 1024),
                    "indexed_at": metadata.get('indexed_at', 'Unknown')
                }
            documents[source]["chunks"] += 1
        
        return {
            "total_chunks": len(all_data['ids']),
            "total_documents": len(documents),
            "total_size_mb": sum(doc['size_mb'] for doc in documents.values()),
            "documents": documents
        }



    
    def clear_all(self) -> bool:
        """Clear all indexed documents"""
        try:
            # Get count before clearing
            count = self.collection.count()
            
            # Delete all
            self.collection.delete(where={})
            
            console.print(f"[green]✓ Cleared {count} chunks from database[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to clear database: {e}[/red]")
            return False
    # Add these methods to the FileParser class in file_parser.py

    def index_directory(self, directory: Path, recursive: bool = True, 
                    force: bool = False, pattern: str = None) -> Dict[str, int]:
        """
        Index all supported files in directory
        
        Args:
            directory: Path to directory
            recursive: Whether to search subdirectories
            force: Re-index already indexed files
            pattern: Optional filename pattern to match (e.g., "*.pdf" or "*atomic*")
        """
        directory = Path(directory).expanduser().resolve()
        
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        
        # Find all supported files
        console.print(f"[cyan]Scanning {directory}...[/cyan]")
        files = []
        
        # Handle iCloud specific paths
        if "Mobile Documents" in str(directory) or "iCloud" in str(directory):
            console.print("[dim]Detected iCloud directory - skipping .icloud placeholder files[/dim]")
        
        for ext in SUPPORTED_EXTENSIONS:
            if recursive:
                search_pattern = f"**/*{ext}"
            else:
                search_pattern = f"*{ext}"
            
            found_files = list(directory.glob(search_pattern))
            
            # Apply additional pattern filter if specified
            if pattern:
                import fnmatch
                found_files = [f for f in found_files if fnmatch.fnmatch(f.name, pattern)]
            
            files.extend(found_files)
        
        # Remove duplicates, hidden files, and iCloud placeholders
        files = sorted(set(f for f in files 
                        if not f.name.startswith('.') 
                        and not f.name.endswith('.icloud')
                        and f.is_file()))
        
        if not files:
            console.print("[yellow]No supported files found[/yellow]")
            if pattern:
                console.print(f"[dim]Pattern filter: {pattern}[/dim]")
            return {"indexed": 0, "skipped": 0, "failed": 0}
        
        console.print(f"[green]Found {len(files)} supported files[/green]")
        
        # Group files by type for summary
        file_types = {}
        for f in files:
            ext = f.suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        console.print("[dim]File types found:[/dim]")
        for ext, count in file_types.items():
            console.print(f"[dim]  {ext}: {count} files[/dim]")
        
        # Process files with progress bar
        stats = {"indexed": 0, "skipped": 0, "failed": 0}
        failed_files = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task(f"Indexing {len(files)} files...", total=len(files))
            
            for file_path in files:
                # Update progress description
                progress.update(task, description=f"Processing: {file_path.name}")
                
                success, message = self.index_file(file_path, force=force)
                
                if success:
                    stats["indexed"] += 1
                elif "Already indexed" in message:
                    stats["skipped"] += 1
                else:
                    stats["failed"] += 1
                    failed_files.append((file_path.name, message))
                
                progress.advance(task)
        
        # Summary report
        console.print("\n[bold green]Indexing Complete![/bold green]")
        console.print(f"✓ Indexed: {stats['indexed']} files")
        console.print(f"⟳ Skipped: {stats['skipped']} files (already indexed)")
        console.print(f"✗ Failed: {stats['failed']} files")
        
        # Show failed files if any
        if failed_files:
            console.print("\n[red]Failed files:[/red]")
            for filename, reason in failed_files[:5]:  # Show first 5
                console.print(f"  • {filename}: {reason}")
            if len(failed_files) > 5:
                console.print(f"  ... and {len(failed_files) - 5} more")
        
        return stats

    def search_by_filename(self, pattern: str) -> List[str]:
        """Search for documents by filename pattern"""
        all_data = self.collection.get()
        matching_files = set()
        
        pattern_lower = pattern.lower()
        
        for metadata in all_data['metadatas']:
            source = metadata.get('source', '')
            # Case-insensitive substring match
            if pattern_lower in source.lower():
                matching_files.add(source)
        
        return sorted(matching_files)

    def remove_document(self, filename: str) -> bool:
        """Remove a specific document from the index"""
        try:
            # First, find all chunks for this document
            # Try exact match first
            results = self.collection.get(where={"source": filename})
            
            if not results['ids']:
                # Try partial match if exact match fails
                all_data = self.collection.get()
                ids_to_delete = []
                
                for i, metadata in enumerate(all_data['metadatas']):
                    if metadata.get('source', '') == filename or filename in metadata.get('source', ''):
                        ids_to_delete.append(all_data['ids'][i])
                
                if not ids_to_delete:
                    console.print(f"[yellow]Document not found: {filename}[/yellow]")
                    console.print("[dim]Available documents:[/dim]")
                    
                    # Show available documents for reference
                    unique_docs = set()
                    for metadata in all_data['metadatas']:
                        unique_docs.add(metadata.get('source', 'Unknown'))
                    
                    for doc in sorted(unique_docs)[:10]:
                        console.print(f"[dim]  • {doc}[/dim]")
                    
                    if len(unique_docs) > 10:
                        console.print(f"[dim]  ... and {len(unique_docs) - 10} more[/dim]")
                    
                    return False
                
                # Delete the chunks
                self.collection.delete(ids=ids_to_delete)
                console.print(f"[green]✓ Removed {len(ids_to_delete)} chunks from {filename}[/green]")
            else:
                # Delete all chunks for exact match
                self.collection.delete(ids=results['ids'])
                console.print(f"[green]✓ Removed {len(results['ids'])} chunks from {filename}[/green]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to remove document: {e}[/red]")
            return False

    def list_documents(self, sort_by: str = "name", filter_pattern: str = None) -> List[Dict]:
        """
        List all indexed documents with details
        """
        all_data = self.collection.get()
        
        if not all_data['ids']:
            return []
        
        # Group by document
        documents = {}
        for metadata in all_data['metadatas']:
            source = metadata.get('source', 'Unknown')
            
            # Apply filter if specified
            if filter_pattern and filter_pattern.lower() not in source.lower():
                continue
            
            if source not in documents:
                # Handle file_size type conversion
                file_size = metadata.get('file_size', 0)
                if isinstance(file_size, str):
                    try:
                        file_size = int(file_size)
                    except ValueError:
                        file_size = 0
                elif file_size is None:
                    file_size = 0
                
                documents[source] = {
                    "name": source,
                    "chunks": 0,
                    "size_mb": file_size / (1024 * 1024),
                    "type": metadata.get('file_type', 'unknown'),
                    "indexed_at": metadata.get('indexed_at', 'Unknown'),
                    "path": metadata.get('full_path', 'Unknown')
                }
            documents[source]["chunks"] += 1
        
        # Convert to list and sort
        doc_list = list(documents.values())
        
        if sort_by == "name":
            doc_list.sort(key=lambda x: x['name'].lower())
        elif sort_by == "size":
            doc_list.sort(key=lambda x: x['size_mb'], reverse=True)
        elif sort_by == "date":
            doc_list.sort(key=lambda x: x['indexed_at'], reverse=True)
        elif sort_by == "chunks":
            doc_list.sort(key=lambda x: x['chunks'], reverse=True)
        
        return doc_list




    def update_directory(self, directory: Path, recursive: bool = True) -> Dict[str, int]:
        """
        Update index with only new or modified files
        """
        directory = Path(directory).expanduser().resolve()
        
        console.print(f"[cyan]Checking for new files in {directory}...[/cyan]")
        
        # Get all currently indexed files
        all_data = self.collection.get()
        indexed_files = {}
        
        for metadata in all_data['metadatas']:
            full_path = metadata.get('full_path')
            file_hash = metadata.get('file_hash')
            if full_path and file_hash:
                indexed_files[full_path] = file_hash
        
        # Find all files in directory
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            if recursive:
                pattern = f"**/*{ext}"
            else:
                pattern = f"*{ext}"
            
            found = directory.glob(pattern)
            files.extend([f for f in found 
                        if f.is_file() 
                        and not f.name.startswith('.')
                        and not f.name.endswith('.icloud')])
        
        # Check which files are new or modified
        new_files = []
        modified_files = []
        
        for file_path in files:
            full_path_str = str(file_path.absolute())
            
            if full_path_str not in indexed_files:
                new_files.append(file_path)
            else:
                # Check if file has been modified
                current_hash = self._get_file_hash(file_path)
                if current_hash != indexed_files[full_path_str]:
                    modified_files.append(file_path)
        
        total_updates = len(new_files) + len(modified_files)
        
        if total_updates == 0:
            console.print("[green]✓ Everything is up to date![/green]")
            return {"indexed": 0, "skipped": 0, "failed": 0}
        
        console.print(f"[yellow]Found {len(new_files)} new files and {len(modified_files)} modified files[/yellow]")
        
        # Index new and modified files
        stats = {"indexed": 0, "skipped": 0, "failed": 0}
        
        all_files = new_files + modified_files
        for file_path in track(all_files, description="Updating index..."):
            success, message = self.index_file(file_path, force=True)
            
            if success:
                stats["indexed"] += 1
                console.print(f"[green]✓ {message}[/green]")
            else:
                stats["failed"] += 1
                console.print(f"[red]✗ {message}[/red]")
        
        return stats

    def watch_directory(self, directory: Path, interval: int = 60):
        """
        Watch directory for changes and auto-index new files
        """
        import time
        
        directory = Path(directory).expanduser().resolve()
        
        console.print(f"[cyan]Watching {directory} for changes...[/cyan]")
        console.print(f"[dim]Checking every {interval} seconds[/dim]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")
        
        try:
            while True:
                # Run update
                stats = self.update_directory(directory)
                
                if stats["indexed"] > 0:
                    console.print(f"[green]Auto-indexed {stats['indexed']} new/modified files[/green]")
                
                # Wait before next check
                time.sleep(interval)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped watching directory[/yellow]")

    def analyze_directory(self, directory: Path) -> Dict:
        """
        Analyze directory without indexing - useful for planning
        """
        directory = Path(directory).expanduser().resolve()
        
        if not directory.exists() or not directory.is_dir():
            console.print(f"[red]Directory not found: {directory}[/red]")
            return {"error": "Directory not found"}
        
        console.print(f"[cyan]Analyzing {directory}...[/cyan]")
        
        # Gather statistics
        stats = {
            "total_files": 0,
            "supported_files": 0,
            "total_size_mb": 0,
            "by_type": {},
            "by_folder": {},
            "largest_files": [],
            "icloud_placeholders": 0
        }
        
        # Find all files
        all_files = list(directory.rglob("*"))
        stats["total_files"] = len([f for f in all_files if f.is_file()])
        
        supported_files = []
        for ext in SUPPORTED_EXTENSIONS:
            files = list(directory.rglob(f"*{ext}"))
            for f in files:
                if f.is_file() and not f.name.startswith('.'):
                    if f.name.endswith('.icloud'):
                        stats["icloud_placeholders"] += 1
                    else:
                        supported_files.append(f)
                        
                        # Count by type
                        stats["by_type"][ext] = stats["by_type"].get(ext, 0) + 1
                        
                        # Count by folder
                        folder = f.parent.name
                        stats["by_folder"][folder] = stats["by_folder"].get(folder, 0) + 1
                        
                        # Track size
                        size_mb = f.stat().st_size / (1024 * 1024)
                        stats["total_size_mb"] += size_mb
                        
                        # Track largest files
                        stats["largest_files"].append((f.name, size_mb))
        
        stats["supported_files"] = len(supported_files)
        stats["largest_files"].sort(key=lambda x: x[1], reverse=True)
        stats["largest_files"] = stats["largest_files"][:10]  # Top 10
        
        # Display analysis
        console.print(f"\n[bold]Directory Analysis:[/bold]")
        console.print(f"Total files: {stats['total_files']}")
        console.print(f"Supported files: {stats['supported_files']}")
        console.print(f"Total size: {stats['total_size_mb']:.1f} MB")
        
        if stats["icloud_placeholders"] > 0:
            console.print(f"[yellow]iCloud placeholders: {stats['icloud_placeholders']} (not downloaded)[/yellow]")
        
        if stats["by_type"]:
            console.print("\n[bold]Files by type:[/bold]")
            for ext, count in sorted(stats["by_type"].items()):
                console.print(f"  {ext}: {count}")
        
        if len(stats["by_folder"]) > 1:
            console.print("\n[bold]Top folders:[/bold]")
            for folder, count in sorted(stats["by_folder"].items(), key=lambda x: x[1], reverse=True)[:5]:
                console.print(f"  {folder}: {count} files")
        
        if stats["largest_files"]:
            console.print("\n[bold]Largest files:[/bold]")
            for name, size in stats["largest_files"][:5]:
                console.print(f"  {name}: {size:.1f} MB")
        
        return stats




if __name__ == "__main__":
    parser = FileParser()
    
    # Test with a specific PDF
    import sys
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        if test_file.exists():
            success, message = parser.index_file(test_file, force=True)
            console.print(f"Result: {message}")
            
            # Show stats
            stats = parser.get_stats()
            console.print(f"\nDatabase stats:")
            console.print(f"Total documents: {stats['total_documents']}")
            console.print(f"Total chunks: {stats['total_chunks']}")


