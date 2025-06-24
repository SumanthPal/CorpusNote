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
from PIL import Image # pip install Pillow
import io
import base64
import cv2  # pip install opencv-python
import google.generativeai as genai
import pytesseract  # pip install pytesseract
from PIL import Image, ExifTags  # Add ExifTags
import easyocr  # Add this import

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
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
            console.print("[green]✓ Gemini vision model initialized[/green]")
        except Exception as e:
            console.print(f"[red]Failed to initialize Gemini: {e}[/red]")
            self.gemini_model = None
        
        try:
            self.easyocr_reader = easyocr.Reader(['en'])
            console.print("[green]✓ EasyOCR initialized[/green]")
        except Exception as e:
            console.print(f"[yellow]EasyOCR not available: {e}[/yellow]")
            self.easyocr_reader = None

        
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
        elif suffix in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp']:
            return self._extract_image_content(file_path)
        elif suffix in CODE_EXTENSIONS:
            return self._extract_code_content(file_path)
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
    
    def _chunk_text(self, text: str, file_name: str, is_image: bool = False) -> List[Dict[str, any]]:
        """
        Smart chunking that preserves context and handles edge cases
        Returns list of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []

        is_code = any(file_name.endswith(ext) for ext in CODE_EXTENSIONS)
        
        if is_image:
            return self._chunk_image_content(text, file_name)
        elif is_code:
            return self._chunk_code(text, file_name)
        
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
    
    def _find_and_filter_files(self, directory: Path, recursive: bool, pattern: Optional[str] = None) -> List[Path]:
        """Finds all supported files in a directory and filters them based on config"""
        
        console.print('[cyan]Scanning {directory} for supported files...[/cyan]')
        all_found_files = []
        for ext in SUPPORTED_EXTENSIONS:
            search_pattern = f"**/*{ext}" if recursive else f"*{ext}"
            all_found_files.extend(directory.glob(search_pattern))
        
        if pattern:
            import fnmatch
            all_found_files = [f for f in all_found_files if fnmatch.fnmatch(f.name, pattern)]
            
        filtered_files = []
        for f in set(all_found_files): # Use set to get unique paths
            if not f.is_file():
                continue

            relative_path = f.relative_to(directory)
            
            # 1. Check against ignored directories
            if any(part in IGNORE_DIRECTORIES for part in relative_path.parts):
                continue
            
            # 2. Check against ignored file patterns
            if any(fnmatch.fnmatch(f.name, glob) for glob in IGNORE_FILES):
                continue
            
            # 3. Check for hidden files and iCloud placeholders
            if f.name.startswith('.') or f.name.endswith('.icloud'):
                continue
            
            filtered_files.append(f)
            
        console.print(f"[green]Found {len(filtered_files)} files to process[/green]")
        return sorted(filtered_files)

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
        
        files = self._find_and_filter_files(directory, recursive, pattern)

        # Handle iCloud specific paths
        if "Mobile Documents" in str(directory) or "iCloud" in str(directory):
            console.print("[dim]Detected iCloud directory - skipping .icloud placeholder files[/dim]")
        
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
        indexed_files = {
            md.get('full_path'): md.get('file_hash')
            for md in all_data['metadatas']
            if md.get('full_path') and md.get('file_hash')
        }
        
        current_files_on_disk = self._find_and_filter_files(directory, recursive)

        
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
    
    def _describe_image(self, file_path: Path) -> str:
        """Generate description of image using Gemini Vision"""
        if not self.gemini_model:
            return ""
        
        try:
            # Load and prepare image
            with Image.open(file_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (Gemini has size limits)
                max_size = 1024
                if max(img.width, img.height) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Create prompt for research diagrams
                prompt = """Analyze this image and provide a detailed description focusing on:
                1. Type of diagram/chart/figure (flowchart, graph, table, etc.)
                2. Main concepts, labels, and text visible
                3. Relationships shown between elements
                4. Any data, numbers, or measurements
                5. Overall purpose or message of the image
                
                Be thorough but concise, focusing on information that would be useful for research."""
                
                # Send to Gemini
                response = self.gemini_model.generate_content([
                    prompt,
                    {"mime_type": "image/jpeg", "data": img_byte_arr}
                ])
                
                return response.text if response.text else ""
                
        except Exception as e:
            console.print(f"[yellow]Failed to describe image {file_path.name}: {e}[/yellow]")
            return ""

    # Image related stuff
    def _preprocess_image(self, image: any) -> any:
        """Enhance image for better OCR results"""
        # FIX: cv1Color should be cvtColor
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Fixed typo
        
        denoised = cv2.fastNlMeansDenoising(gray)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
   
        
    def _extract_image_content(self, file_path: Path) -> str:
        """Extract all content from image file"""
        if not self._check_image_quality(file_path):
            console.print(f"[yellow]Skipping low-quality image: {file_path.name}[/yellow]")
            return ""
        
        content_parts = []
        
        # Extract text via OCR
        console.print(f"[dim]Extracting text from {file_path.name}...[/dim]")
        ocr_text = self._extract_image_text(file_path)
        if ocr_text.strip():
            content_parts.append(f"[OCR Text]\n{ocr_text}")
        
        # Get image description using Gemini Vision
        console.print(f"[dim]Analyzing image content...[/dim]")
        description = self._describe_image(file_path)
        if description.strip():
            content_parts.append(f"[Image Description]\n{description}")
        
        # Add basic metadata as searchable text
        try:
            with Image.open(file_path) as img:
                meta_info = f"Image dimensions: {img.width}x{img.height}, Format: {img.format}"
                content_parts.append(f"[Image Info]\n{meta_info}")
        except:
            pass
        
        final_content = "\n\n".join(content_parts)
        
        if not final_content.strip():
            console.print(f"[yellow]No content extracted from {file_path.name}[/yellow]")
            return ""
        
        console.print(f"[green]Extracted {len(final_content)} characters from {file_path.name}[/green]")
        return final_content
    
    def _extract_image_metadata(self, file_path: Path) -> Dict:
        """Extract metadata from image"""
        try:
            with Image.open(file_path) as img:
                # Basic info
                metadata = {
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'mode': img.mode,
                    'file_size': file_path.stat().st_size
                }
                
                # EXIF data
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    for tag_id, value in exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        if isinstance(value, str) and len(value) < 100:  # Keep only short text values
                            metadata[f'exif_{tag}'] = value
                
                return metadata
        except Exception as e:
            console.print(f"[yellow]Failed to extract metadata from {file_path.name}: {e}[/yellow]")
            return {}
    
    def _chunk_image_content(self, content: str, file_name: str) -> List[Dict[str, any]]:
        """Special chunking for image content - keep as single chunk"""
        if not content.strip():
            return []
        
        # For images, create one comprehensive chunk
        chunk = {
            'text': content.strip(),
            'page': 'Image',
            'chunk_index': 0,
            'content_type': 'image'
        }
        
        console.print(f"[green]Created 1 image chunk from {file_name}[/green]")
        return [chunk]
            
    def _create_image_metadata(self, file_path: Path, chunk_data: Dict) -> Dict:
        """Create metadata specific to image files"""
        base_metadata = {
            "source": file_path.name,
            "full_path": str(file_path.absolute()),
            "file_hash": self._get_file_hash(file_path),
            "chunk_index": chunk_data.get('chunk_index', 0),
            "file_size": file_path.stat().st_size,
            "file_type": file_path.suffix.lower(),
            "indexed_at": datetime.now().isoformat(),
            "content_type": "image"
        }
        
        # Add image-specific metadata
        img_metadata = self._extract_image_metadata(file_path)
        base_metadata.update(img_metadata)
        
        return self._safe_metadata(base_metadata)
    
    def _get_dominant_colors(self, img, num_colors=3):
        """Extract dominant colors from image"""
        try:
            # Convert to RGB and resize for faster processing
            img_rgb = img.convert('RGB')
            img_small = img_rgb.resize((50, 50))
            
            # Get color data
            colors = img_small.getcolors(maxcolors=256*256*256)
            if colors:
                # Sort by frequency and get top colors
                colors.sort(key=lambda x: x[0], reverse=True)
                dominant = [f"rgb{color[1]}" for color in colors[:num_colors]]
                return dominant
        except Exception:
            pass
        return []
    
    def _check_image_quality(self, file_path: Path) -> bool:
        """Check if image is suitable for OCR"""
        try:
            with Image.open(file_path) as img:
                # Skip very small images
                if img.width < 100 or img.height < 100:
                    return False
                
                # Skip very large images (resize them first)
                if img.width * img.height > 4000 * 4000:
                    console.print(f"[yellow]Large image {file_path.name}, may take longer to process[/yellow]")
                
                return True
        except Exception:
            return False
        
    
        
    def _extract_image_text(self, file_path: Path) -> str:
        """Extract text from image using multiple OCR approaches"""
        try:
            # Load image
            image = cv2.imread(str(file_path))
            if image is None:
                console.print(f"[red]Could not load image: {file_path}[/red]")
                return ""
            
            # Try multiple preprocessing approaches
            results = []
            
            # 1. Original image
            if self.easyocr_reader:
                try:
                    easyocr_results = self.easyocr_reader.readtext(image)
                    easyocr_text = ' '.join([result[1] for result in easyocr_results if result[2] > 0.3])
                    if easyocr_text:
                        results.append(easyocr_text)
                except Exception as e:
                    console.print(f"[yellow]EasyOCR failed: {e}[/yellow]")
            
            # 2. Preprocessed image
            processed_image = self._preprocess_image(image)
            tesseract_text = pytesseract.image_to_string(processed_image, config='--psm 6')
            if tesseract_text.strip():
                results.append(tesseract_text)
            
            # 3. Try with different PSM modes for diagrams
            psm_modes = [3, 4, 6, 8, 11, 12]  # Different page segmentation modes
            for psm in psm_modes:
                try:
                    text = pytesseract.image_to_string(processed_image, config=f'--psm {psm}')
                    if text.strip() and len(text) > len(tesseract_text):
                        results.append(text)
                        break
                except:
                    continue
            
            # Combine results
            combined_text = '\n'.join(results) if results else ""
            return self._clean_text(combined_text)
            
        except Exception as e:
            console.print(f"[red]OCR extraction failed for {file_path.name}: {e}[/red]")
            return ""
        
    
    # ---- CODE PARSING ----
    def _detect_language(self, file_path: Path) -> str:
        ext_to_lang = { '.py': 'Python', 
                       '.js': 'JavaScript', 
                       '.ts': 'TypeScript', 
                       '.java': 'Java', 
                       '.cpp': 'C++', 
                       '.c': 'C', 
                       '.h': 'C/C++ Header',
                       '.hpp': 'C++ Header', 
                       '.go': 'Go', 
                       '.rs': 'Rust', 
                       '.rb': 'Ruby', 
                       '.php': 'PHP', 
                       '.swift': 'Swift', 
                       '.kt': 'Kotlin', 
                       '.scala': 'Scala', 
                       '.r': 'R',
                       '.m': 'MATLAB/Objective-C', 
                       '.cs': 'C#', 
                       '.sql': 'SQL', 
                       '.sh': 'Shell', 
                       '.bash': 'Bash', 
                       '.zsh': 'Zsh', 
                       '.fish': 'Fish', 
                       '.jsx':'JavaScript/React', 
                       '.tsx': 'TypeScript/React', 
                       '.vue': 'Vue', 
                       '.yaml': 'YAML', 
                       '.yml': 'YAML', 
                       '.json': 'JSON', 
                       '.xml': 'XML',
                       '.html': 'HTML', 
                       '.css': 'CSS', 
                       '.scss': 'SCSS', 
                       '.less': 'LESS', 
                       '.dockerfile': 'Dockerfile', 
                       '.makefile': 'Makefile', 
                       '.cmake':'CMake', 
                       '.gradle': 'Gradle' 
                       }
        return ext_to_lang.get(file_path.suffix.lower(), 'Unknown')
    
    def _extract_code_structure(self, code: str, file_path: Path) -> str:
        """Extract high-level structure (classes, functions, etc.)"""
        language = self._detect_language(file_path)
        structure_parts = []
        
        try:
            if language == 'Python':
                import re
                classes = re.findall(r'^class\s+(\w+)', code, re.MULTILINE)
                functions = re.findall(r'^def\s+(\w+)', code, re.MULTILINE)
                
                if classes:
                    structure_parts.append(f"Classes: {', '.join(classes[:10])}")
                if functions:
                    # Filter out common Python methods
                    filtered_funcs = [f for f in functions if not f.startswith('_') or f in ['__init__', '__str__', '__repr__']]
                    if filtered_funcs:
                        structure_parts.append(f"Functions: {', '.join(filtered_funcs[:15])}")
                        
            elif language in ['JavaScript', 'TypeScript', 'JavaScript/React', 'TypeScript/React']:
                import re
                # ES6 classes and functions
                classes = re.findall(r'class\s+(\w+)', code)
                functions = re.findall(r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:\([^)]*\)|async))', code)
                functions = [f[0] or f[1] for f in functions if f[0] or f[1]]
                
                if classes:
                    structure_parts.append(f"Classes: {', '.join(classes[:10])}")
                if functions:
                    structure_parts.append(f"Functions: {', '.join(functions[:15])}")
                    
            elif language == 'Java':
                import re
                classes = re.findall(r'(?:public\s+)?(?:class|interface|enum)\s+(\w+)', code)
                methods = re.findall(r'(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*{', code)
                
                if classes:
                    structure_parts.append(f"Classes/Interfaces: {', '.join(classes[:10])}")
                if methods:
                    structure_parts.append(f"Methods: {', '.join(methods[:15])}")
                    
            elif language in ['C++', 'C']:
                import re
                # Simple function detection
                functions = re.findall(r'(?:^|\n)\s*(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*{', code, re.MULTILINE)
                if functions:
                    structure_parts.append(f"Functions: {', '.join(functions[:15])}")
                    
            elif language == 'Go':
                import re
                functions = re.findall(r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(', code)
                types = re.findall(r'type\s+(\w+)\s+(?:struct|interface)', code)
                
                if types:
                    structure_parts.append(f"Types: {', '.join(types[:10])}")
                if functions:
                    structure_parts.append(f"Functions: {', '.join(functions[:15])}")
            
            elif language == 'Rust':
                import re
                functions = re.findall(r'fn\s+(\w+)', code)
                structs = re.findall(r'struct\s+(\w+)', code)
                enums = re.findall(r'enum\s+(\w+)', code)
                traits = re.findall(r'trait\s+(\w+)', code)
                
                if structs or enums:
                    structure_parts.append(f"Types: {', '.join((structs + enums)[:10])}")
                if traits:
                    structure_parts.append(f"Traits: {', '.join(traits[:5])}")
                if functions:
                    structure_parts.append(f"Functions: {', '.join(functions[:15])}")

            elif language == 'Ruby':
                import re
                classes = re.findall(r'class\s+(\w+)', code)
                modules = re.findall(r'module\s+(\w+)', code)
                methods = re.findall(r'def\s+(\w+)', code)
                
                if classes or modules:
                    structure_parts.append(f"Classes/Modules: {', '.join((classes + modules)[:10])}")
                if methods:
                    structure_parts.append(f"Methods: {', '.join(methods[:15])}")

            elif language == 'PHP':
                import re
                classes = re.findall(r'class\s+(\w+)', code)
                functions = re.findall(r'function\s+(\w+)', code)
                traits = re.findall(r'trait\s+(\w+)', code)
                
                if classes:
                    structure_parts.append(f"Classes: {', '.join(classes[:10])}")
                if traits:
                    structure_parts.append(f"Traits: {', '.join(traits[:5])}")
                if functions:
                    structure_parts.append(f"Functions: {', '.join(functions[:15])}")

            elif language == 'C#':
                import re
                classes = re.findall(r'(?:public\s+|private\s+|internal\s+)?(?:class|interface|struct)\s+(\w+)', code)
                methods = re.findall(r'(?:public|private|protected|internal)?\s*(?:static\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)', code)
                
                if classes:
                    structure_parts.append(f"Types: {', '.join(classes[:10])}")
                if methods:
                    # Filter out common methods
                    filtered = [m for m in methods if m not in ['Main', 'ToString', 'GetHashCode', 'Equals']]
                    if filtered:
                        structure_parts.append(f"Methods: {', '.join(filtered[:15])}")

            elif language == 'Swift':
                import re
                classes = re.findall(r'(?:class|struct|enum|protocol)\s+(\w+)', code)
                functions = re.findall(r'func\s+(\w+)', code)
                
                if classes:
                    structure_parts.append(f"Types: {', '.join(classes[:10])}")
                if functions:
                    structure_parts.append(f"Functions: {', '.join(functions[:15])}")

            elif language == 'Kotlin':
                import re
                classes = re.findall(r'(?:class|interface|object|data\s+class)\s+(\w+)', code)
                functions = re.findall(r'fun\s+(\w+)', code)
                
                if classes:
                    structure_parts.append(f"Types: {', '.join(classes[:10])}")
                if functions:
                    structure_parts.append(f"Functions: {', '.join(functions[:15])}")

            elif language == 'Scala':
                import re
                classes = re.findall(r'(?:class|trait|object|case\s+class)\s+(\w+)', code)
                functions = re.findall(r'def\s+(\w+)', code)
                
                if classes:
                    structure_parts.append(f"Types: {', '.join(classes[:10])}")
                if functions:
                    structure_parts.append(f"Functions: {', '.join(functions[:15])}")

            elif language == 'R':
                import re
                functions = re.findall(r'(\w+)\s*<-\s*function', code)
                functions2 = re.findall(r'(\w+)\s*=\s*function', code)
                
                all_functions = list(set(functions + functions2))
                if all_functions:
                    structure_parts.append(f"Functions: {', '.join(all_functions[:15])}")

            elif language == 'SQL':
                import re
                tables = re.findall(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', code, re.IGNORECASE)
                views = re.findall(r'CREATE\s+VIEW\s+(\w+)', code, re.IGNORECASE)
                procedures = re.findall(r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+(\w+)', code, re.IGNORECASE)
                
                if tables:
                    structure_parts.append(f"Tables: {', '.join(tables[:10])}")
                if views:
                    structure_parts.append(f"Views: {', '.join(views[:10])}")
                if procedures:
                    structure_parts.append(f"Procedures: {', '.join(procedures[:10])}")

            elif language in ['Shell', 'Bash', 'Zsh', 'Fish']:
                import re
                functions = re.findall(r'(?:function\s+)?(\w+)\s*\(\)', code)
                
                if functions:
                    structure_parts.append(f"Functions: {', '.join(functions[:15])}")



                    
        except Exception as e:
            console.print(f"[yellow]Could not extract structure from {file_path.name}: {e}[/yellow]")
        
        return '; '.join(structure_parts) if structure_parts else ""
    
    def _extract_code_content(self, file_path: Path) -> str:
        """Extract and enrich code file with metadata"""
        
        try:
            code = self._extract_plain_text(file_path)
            if not code:
                return ""
            
            language = self._detect_language(file_path)
            lines = code.splitlines()
            line_count = len(lines)
            
            structure = self._extract_code_structure(code, file_path)
            
            imports = []
            if language == 'Python':
                imports = [line.strip() for line in lines if line.strip().startswith(('import ', 'from '))][:5]
            elif language in ['JavaScript', 'TypeScript', 'JavaScript/React', 'TypeScript/React']:
                imports = [line.strip() for line in lines if line.strip().startswith(('import ', 'require('))[:5]]
            elif language == 'Java':
                imports = [line.strip() for line in lines if line.strip().startswith('import ')][:5]
            elif language in ['C++', 'C']:
                imports = [line.strip() for line in lines if line.strip().startswith('#include')][:5]
            elif language == 'Go':
                imports = [line.strip() for line in lines if line.strip().startswith('import ')][:5]
            elif language == 'Rust':
                imports = [line.strip() for line in lines if line.strip().startswith('use ')][:5]
            elif language == 'Ruby':
                imports = [line.strip() for line in lines if line.strip().startswith('require ')][:5]
            elif language == 'PHP':
                imports = [line.strip() for line in lines if line.strip().startswith('use ')][:5]
            elif language == 'C#':
                imports = [line.strip() for line in lines if line.strip().startswith('using ')][:5]
            
            # TODO:  ADD MORE LANGUAGE SUPPORT
            
            content_parts = [
                f'[Code File: {file_path.name}]',
                f'[Language: {language}]',
                f'[Lines: {line_count}]',
            ]

            if structure:
                content_parts.append(f'[Structure: {structure}]')
            
            if imports:
                content_parts.append(f'[Key Imports: {", ".join(imports)}]')
            
            content_parts.extend(["", "--- CODE ---", code])
            return "\n".join(content_parts)
        
        except Exception as e:
            console.print(f"[red]Failed to extract code content from {file_path.name}: {e}[/red]")
            return ""
        
    def _chunk_code(self, text: str, file_name: str) -> List[Dict[str, any]]:
        """Special chunking for code files"""
        if not text.strip():
            return []
        
        # Extract metadata section and code section
        lines = text.split('\n')
        code_start = 0
        
        # Find where actual code starts (after metadata)
        for i, line in enumerate(lines):
            if line.strip() == "--- CODE ---":
                code_start = i + 1
                break
        
        metadata_section = '\n'.join(lines[:code_start-1])
        code_section = '\n'.join(lines[code_start:])
        
        # For code files, we'll chunk more conservatively
        # Try to keep functions/classes together
        chunks = []
        
        # If the file is small enough, keep it as one chunk
        if len(text) < CHUNK_SIZE * 150:  # ~3000 words
            chunks.append({
                'text': text.strip(),
                'page': 'Code',
                'chunk_index': 0,
                'content_type': 'code'
            })
            console.print(f"[green]Created 1 code chunk from {file_name}[/green]")
            return chunks
        
        # Otherwise, chunk by logical sections
        current_chunk = [metadata_section, ""]  # Always include metadata
        current_size = len(metadata_section.split())
        
        code_lines = code_section.split('\n')
        in_function = False
        function_depth = 0
        
        for line in code_lines:
            stripped = line.strip()
            
            # Simple heuristic to detect function/class boundaries
            if any(keyword in stripped for keyword in ['def ', 'class ', 'function ', 'func ', 'public ', 'private ']):
                in_function = True
                function_depth = 0
            
            # Track braces/indentation to know when function ends
            if in_function:
                function_depth += line.count('{') - line.count('}')
                if stripped and not line.startswith((' ', '\t')) and function_depth == 0:
                    in_function = False
            
            # Check if adding this line would exceed chunk size
            line_words = len(line.split())
            if current_size + line_words > CHUNK_SIZE and not in_function and current_chunk:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text.strip(),
                    'page': 'Code',
                    'chunk_index': len(chunks),
                    'content_type': 'code'
                })
                
                # Start new chunk with metadata and some overlap
                current_chunk = [metadata_section, ""]
                if len(chunks) > 0 and CHUNK_OVERLAP > 0:
                    # Add last few lines as overlap
                    overlap_lines = code_lines[max(0, code_lines.index(line) - 10):code_lines.index(line)]
                    current_chunk.extend(overlap_lines)
                
                current_size = len(' '.join(current_chunk).split())
            
            current_chunk.append(line)
            current_size += line_words
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) > 2:  # More than just metadata
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text.strip(),
                'page': 'Code',
                'chunk_index': len(chunks),
                'content_type': 'code'
            })
        
        console.print(f"[green]Created {len(chunks)} code chunks from {file_name}[/green]")
        return chunk_text

        

                
    
    
        


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


