import requests
from bs4 import BeautifulSoup
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re
import time
from urllib.parse import urlparse, urljoin
from github import Github
import base64
import markdown2
from config import *

console = Console()

class WebParser:
    def __init__(self, file_parser):
        """Initialize web parser with reference to file parser for indexing"""
        self.file_parser = file_parser
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'
        })
        
    def parse_webpage(self, url: str) -> Tuple[str, Dict]:
        """Parse a single webpage and return content + metadata"""
        try:
            console.print(f"[cyan]Fetching {url}...[/cyan]")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract metadata
            metadata = {
                'url': url,
                'title': soup.find('title').text if soup.find('title') else urlparse(url).netloc,
                'description': '',
                'content_type': 'webpage'
            }
            
            # Try to get description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                metadata['description'] = meta_desc.get('content', '')
            
            # Extract main content
            # Try different content containers
            main_content = None
            for selector in ['main', 'article', '[role="main"]', '#content', '.content', '#main', '.main']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.find('body')
            
            if not main_content:
                return "", metadata
            
            # Extract text with some structure preservation
            text_parts = []
            
            # Add title
            text_parts.append(f"[Webpage: {metadata['title']}]")
            text_parts.append(f"[URL: {url}]")
            if metadata['description']:
                text_parts.append(f"[Description: {metadata['description']}]")
            text_parts.append("")
            
            # Extract headings and paragraphs with structure
            for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'td', 'th']):
                text = element.get_text(strip=True)
                if text:
                    if element.name.startswith('h'):
                        level = int(element.name[1])
                        prefix = '#' * level
                        text_parts.append(f"\n{prefix} {text}\n")
                    else:
                        text_parts.append(text)
            
            content = '\n'.join(text_parts)
            
            # Clean up excessive whitespace
            content = re.sub(r'\n{3,}', '\n\n', content)
            
            return content, metadata
            
        except Exception as e:
            console.print(f"[red]Error parsing {url}: {e}[/red]")
            return "", {'url': url, 'error': str(e)}
    
    def parse_website(self, url: str, max_pages: int = 50, same_domain_only: bool = True) -> List[Tuple[str, Dict]]:
        """Parse multiple pages from a website"""
        parsed_urls = set()
        to_parse = [url]
        results = []
        base_domain = urlparse(url).netloc
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Parsing website...", total=max_pages)
            
            while to_parse and len(results) < max_pages:
                current_url = to_parse.pop(0)
                
                if current_url in parsed_urls:
                    continue
                
                parsed_urls.add(current_url)
                progress.update(task, description=f"Parsing: {current_url[:50]}...")
                
                content, metadata = self.parse_webpage(current_url)
                
                if content:
                    results.append((content, metadata))
                    progress.advance(task)
                    
                    # Extract links for crawling
                    if len(results) < max_pages:
                        try:
                            response = self.session.get(current_url, timeout=30)
                            soup = BeautifulSoup(response.content, 'html.parser')
                            
                            for link in soup.find_all('a', href=True):
                                href = link['href']
                                full_url = urljoin(current_url, href)
                                
                                # Check if we should parse this URL
                                if same_domain_only and urlparse(full_url).netloc != base_domain:
                                    continue
                                
                                if full_url not in parsed_urls and full_url not in to_parse:
                                    # Skip common non-content URLs
                                    if not any(skip in full_url.lower() for skip in [
                                        'login', 'signup', 'register', '.pdf', '.jpg', '.png',
                                        '.zip', '.exe', 'mailto:', 'javascript:', '#'
                                    ]):
                                        to_parse.append(full_url)
                        except:
                            pass
                
                # Be polite
                time.sleep(0.5)
        
        console.print(f"[green]Parsed {len(results)} pages from {base_domain}[/green]")
        return results
    
    def parse_github_repo(self, repo_url: str, github_token: Optional[str] = None) -> List[Tuple[str, Dict]]:
        """Parse a GitHub repository"""
        try:
            # Extract owner and repo name from URL
            # Handle URLs like: https://github.com/owner/repo
            parts = repo_url.rstrip('/').split('/')
            if 'github.com' not in repo_url or len(parts) < 5:
                raise ValueError("Invalid GitHub URL. Expected format: https://github.com/owner/repo")
            
            owner = parts[-2]
            repo_name = parts[-1]
            
            console.print(f"[cyan]Accessing GitHub repo: {owner}/{repo_name}[/cyan]")
            
            # Initialize GitHub client
            g = Github(github_token) if github_token else Github()
            repo = g.get_repo(f"{owner}/{repo_name}")
            
            results = []
            
            # 1. Add README
            try:
                readme = repo.get_readme()
                content = base64.b64decode(readme.content).decode('utf-8')
                
                # Convert markdown to structured text
                if readme.name.lower().endswith('.md'):
                    # Keep as markdown for better structure
                    formatted_content = f"[GitHub Repository: {repo.full_name}]\n[File: {readme.name}]\n\n{content}"
                else:
                    formatted_content = f"[GitHub Repository: {repo.full_name}]\n[File: {readme.name}]\n\n{content}"
                
                metadata = {
                    'url': readme.html_url,
                    'title': f"{repo.full_name} - {readme.name}",
                    'path': readme.path,
                    'content_type': 'github_file',
                    'repo': repo.full_name
                }
                
                results.append((formatted_content, metadata))
                console.print(f"[green]✓ Indexed README[/green]")
            except:
                console.print(f"[yellow]No README found[/yellow]")
            
            # 2. Parse repository structure and code files
            console.print(f"[cyan]Scanning repository contents...[/cyan]")
            
            # Get all files in repo
            contents = repo.get_contents("")
            files_to_process = []
            
            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    # Don't go into common non-code directories
                    if file_content.path not in ['node_modules', '.git', 'vendor', 'build', 'dist', '__pycache__']:
                        try:
                            contents.extend(repo.get_contents(file_content.path))
                        except:
                            pass
                else:
                    # Check if it's a supported code file
                    if any(file_content.name.endswith(ext) for ext in CODE_EXTENSIONS):
                        files_to_process.append(file_content)
            
            # Process code files (limit to avoid rate limits)
            max_files = 50
            console.print(f"[dim]Found {len(files_to_process)} code files, processing up to {max_files}[/dim]")
            
            for i, file_content in enumerate(files_to_process[:max_files]):
                try:
                    # Skip large files
                    if file_content.size > 1000000:  # 1MB
                        continue
                    
                    content = base64.b64decode(file_content.content).decode('utf-8')
                    
                    formatted_content = f"[GitHub Repository: {repo.full_name}]\n[File: {file_content.path}]\n\n{content}"
                    
                    metadata = {
                        'url': file_content.html_url,
                        'title': f"{repo.full_name} - {file_content.path}",
                        'path': file_content.path,
                        'content_type': 'github_code',
                        'repo': repo.full_name,
                        'language': self.file_parser._detect_language(Path(file_content.name))
                    }
                    
                    results.append((formatted_content, metadata))
                    
                    if (i + 1) % 10 == 0:
                        console.print(f"[dim]Processed {i + 1} files...[/dim]")
                    
                except Exception as e:
                    console.print(f"[yellow]Skipped {file_content.path}: {e}[/yellow]")
            
            # 3. Add repository metadata
            repo_info = f"""[GitHub Repository: {repo.full_name}]
[Repository Information]

Name: {repo.name}
Owner: {repo.owner.login}
Description: {repo.description or 'No description'}
Language: {repo.language or 'Multiple'}
Stars: {repo.stargazers_count}
Forks: {repo.forks_count}
Created: {repo.created_at}
Updated: {repo.updated_at}
Topics: {', '.join(repo.get_topics()) if repo.get_topics() else 'None'}

URL: {repo.html_url}
"""
            
            metadata = {
                'url': repo.html_url,
                'title': f"{repo.full_name} - Repository Info",
                'content_type': 'github_repo_info',
                'repo': repo.full_name
            }
            
            results.append((repo_info, metadata))
            
            console.print(f"[green]✓ Indexed {len(results)} items from {repo.full_name}[/green]")
            return results
            
        except Exception as e:
            console.print(f"[red]Error parsing GitHub repo: {e}[/red]")
            return []
    
    def index_webpage(self, url: str) -> bool:
        """Index a single webpage"""
        content, metadata = self.parse_webpage(url)
        
        if not content:
            console.print(f"[red]Failed to extract content from {url}[/red]")
            return False
        
        # Create a pseudo-file path for the webpage
        safe_filename = re.sub(r'[^\w\-_\.]', '_', metadata.get('title', 'webpage'))
        pseudo_path = Path(f"/web/{safe_filename}.html")
        
        # Use file parser's chunking logic
        chunks = self.file_parser._chunk_text(content, pseudo_path.name)
        
        if not chunks:
            console.print(f"[red]No valid chunks created from {url}[/red]")
            return False
        
        # Prepare for ChromaDB
        file_hash = str(hash(url))  # Simple hash for web content
        ids = [f"web_{file_hash}_{i}" for i in range(len(chunks))]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "source": metadata.get('title', url),
                "full_path": url,
                "file_hash": file_hash,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "page": chunk.get('page', 'Web'),
                "file_type": '.html',
                "content_type": metadata.get('content_type', 'webpage'),
                "url": url,
                "indexed_at": datetime.now().isoformat()
            }
            
            if metadata.get('description'):
                chunk_metadata['description'] = metadata['description']
            
            metadatas.append(self.file_parser._safe_metadata(chunk_metadata))
        
        # Add to collection
        self.file_parser.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        
        console.print(f"[green]✓ Indexed {url} ({len(chunks)} chunks)[/green]")
        return True
    
    def index_website(self, url: str, max_pages: int = 50, same_domain_only: bool = True) -> Dict[str, int]:
        """Index multiple pages from a website"""
        pages = self.parse_website(url, max_pages, same_domain_only)
        
        stats = {"indexed": 0, "failed": 0}
        
        for content, metadata in pages:
            # Similar to index_webpage but with the already parsed content
            safe_filename = re.sub(r'[^\w\-_\.]', '_', metadata.get('title', 'webpage'))
            pseudo_path = Path(f"/web/{safe_filename}.html")
            
            chunks = self.file_parser._chunk_text(content, pseudo_path.name)
            
            if chunks:
                file_hash = str(hash(metadata['url']))
                ids = [f"web_{file_hash}_{i}" for i in range(len(chunks))]
                documents = [chunk['text'] for chunk in chunks]
                metadatas = []
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        "source": metadata.get('title', metadata['url']),
                        "full_path": metadata['url'],
                        "file_hash": file_hash,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "page": chunk.get('page', 'Web'),
                        "file_type": '.html',
                        "content_type": 'webpage',
                        "url": metadata['url'],
                        "indexed_at": datetime.now().isoformat()
                    }
                    metadatas.append(self.file_parser._safe_metadata(chunk_metadata))
                
                self.file_parser.collection.add(
                    documents=documents,
                    ids=ids,
                    metadatas=metadatas
                )
                stats["indexed"] += 1
            else:
                stats["failed"] += 1
        
        return stats
    
    def index_github_repo(self, repo_url: str, github_token: Optional[str] = None) -> Dict[str, int]:
        """Index a GitHub repository"""
        items = self.parse_github_repo(repo_url, github_token)
        
        stats = {"indexed": 0, "failed": 0}
        
        for content, metadata in items:
            # Create appropriate pseudo-path based on content type
            if metadata['content_type'] == 'github_code':
                pseudo_path = Path(metadata['path'])
            else:
                safe_filename = re.sub(r'[^\w\-_\.]', '_', metadata.get('title', 'github'))
                pseudo_path = Path(f"/github/{safe_filename}.md")
            
            # Use appropriate chunking based on content type
            if metadata['content_type'] == 'github_code' and any(str(pseudo_path).endswith(ext) for ext in CODE_EXTENSIONS):
                chunks = self.file_parser._chunk_code(content, pseudo_path.name)
            else:
                chunks = self.file_parser._chunk_text(content, pseudo_path.name)
            
            if chunks:
                file_hash = str(hash(metadata['url']))
                ids = [f"github_{file_hash}_{i}" for i in range(len(chunks))]
                documents = [chunk['text'] for chunk in chunks]
                metadatas = []
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        "source": metadata.get('title', metadata['url']),
                        "full_path": metadata['url'],
                        "file_hash": file_hash,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "page": chunk.get('page', 'GitHub'),
                        "file_type": pseudo_path.suffix or '.md',
                        "content_type": metadata['content_type'],
                        "url": metadata['url'],
                        "github_repo": metadata.get('repo', ''),
                        "github_path": metadata.get('path', ''),
                        "indexed_at": datetime.now().isoformat()
                    }
                    
                    if metadata.get('language'):
                        chunk_metadata['language'] = metadata['language']
                    
                    metadatas.append(self.file_parser._safe_metadata(chunk_metadata))
                
                self.file_parser.collection.add(
                    documents=documents,
                    ids=ids,
                    metadatas=metadatas
                )
                stats["indexed"] += 1
            else:
                stats["failed"] += 1
        
        return stats


