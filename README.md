# Corpus CLI 📚

A powerful command-line tool for indexing, searching, and chatting with your documents using AI. Features include document indexing, AI-powered chat, web scraping, and automated diagram generation.

## Features ✨

- **Document Indexing**: Index PDFs, text files, markdown, and more with multi-threaded processing
- **AI Chat Interface**: Chat with your documents using Google's Gemini AI
- **Web Indexing**: Index websites and GitHub repositories
- **Diagram Generation**: Create beautiful diagrams from natural language or document content
- **Smart Search**: Vector-based semantic search across all indexed content
- **Export & Analysis**: Export conversations, analyze document collections

## Installation 🚀

### Quick Install (Recommended)

#### Linux/macOS:
```bash
git clone https://github.com/SumanthPal/CorpusNote.git
cd CorpusNote
chmod +x installation/install.sh
./installation/install.sh

```

#### Windows:
```powershell
git clone https://github.com/SumanthPal/CorpusNote.git
cd corpus-cli
powershell -ExecutionPolicy Bypass -File installation\install.ps1
```

#### Using Make:
```bash
git clone https://github.com/SumanthPal/CorpusNote.git
cd corpus-cli
make -f installation/Makefile install
```

The installation script will:
- ✅ Check Python version (3.8+ required)
- ✅ Create a virtual environment
- ✅ Install all dependencies
- ✅ Set up the `corpus` command
- ✅ Check for optional tools (D2)
- ✅ Guide you through initial configuration

### Manual Installation

If you prefer to install manually:

```bash
# Clone repository
git clone https://github.com/SumanthPal/CorpusNote.git
cd CorpusNote

# Create virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install corpus command
pip install -e .

# Configure
corpus config setup
```

### Post-Installation


**⚠️ IMPORTANT: Always activate the virtual environment first!**

```bash
# Activate virtual environment (required every time)
source env/bin/activate  # Linux/macOS
# or
env\Scripts\activate     # Windows
```
```
# Now you can use corpus!
corpus --help
corpus config setup

```

### Making Corpus Available Globally

After installation, you have several options to make the `corpus` comamdn easily accessible.

#### Option 1: Auto-configure Shell Alias (Recommended)

Run the provided script to automatically add corpus to your shell:

```bash
cd CorpusNote
./scripts/add_corpus_alias.sh
source ~/.zshrc  # or ~/.bashrc
```

This creates a shell function that automatically activates the environment when you run corpus.

#### Option 2: Manual Shell Alias
Add this to your `~/.zshrc` or `~/.bashrc`:
```bash
# Corpus CLI
corpus() {
    cd /path/to/CorpusNote && source env/bin/activate && command corpus "$@"
    cd - > /dev/null
}
```

#### Option 3: Global Wrapper Script
From system-wide access without shell configuration.
```bash
sudo cp installation/corpus-wrapper.sh /usr/local/bin/corpus
sudo chmod +x /usr/local/bin/corpus
```

## Quick Start 🏃‍♂️

### Index your documents
```bash
# Index a single file
corpus index ~/Documents/paper.pdf

# Index a directory recursively
corpus index ~/Documents/Research

# Index with specific pattern
corpus index ~/Documents --pattern "*.pdf"

# Index with custom thread count
corpus index ~/Documents --workers 8
```

### Chat with your documents
```bash
# Start interactive chat
corpus chat

# Ask a single question
corpus ask "What are the main findings about quantum computing?"

# Chat with filtered documents
corpus chat --filter "quantum*.pdf"
```

### Generate diagrams
```bash
# Interactive diagram mode
corpus diagram

# Generate diagram from description
corpus diagram "user authentication flow" --type flowchart

# Generate from document search
corpus diagram --search "network architecture" --theme professional

# Batch generation
corpus diagram-batch --file queries.txt
```

## Command Reference 📖

### Document Management

#### `index` - Index documents
```bash
corpus index <path> [OPTIONS]

Options:
  -r, --recursive/--no-recursive  Recursively index subdirectories [default: True]
  -f, --force                     Force re-indexing of existing files
  -p, --pattern TEXT              File pattern to match (e.g., '*.pdf')
  -w, --workers INTEGER           Number of threads (default: auto-detect)
  --no-threading                  Disable multi-threading
```

#### `update` - Update existing index
```bash
corpus update <directory> [OPTIONS]

Options:
  -r, --recursive/--no-recursive  Recursively check subdirectories
  -w, --workers INTEGER           Number of threads
```

#### `watch` - Auto-index on changes
```bash
corpus watch <directory> [OPTIONS]

Options:
  -i, --interval INTEGER  Check interval in seconds [default: 60]
```

### Web Indexing

#### `index-url` - Index websites
```bash
corpus index-url <url> [OPTIONS]

Options:
  -m, --max-pages INTEGER         Maximum pages to crawl [default: 50]
  --same-domain/--any-domain      Only crawl same domain [default: True]
  -t, --github-token TEXT         GitHub personal access token
```

#### `index-github` - Index GitHub repos
```bash
corpus index-github <owner/repo> [OPTIONS]

Options:
  -t, --token TEXT  GitHub personal access token for private repos
```

### Chat & Query

#### `chat` - Interactive chat
```bash
corpus chat [OPTIONS]

Options:
  -f, --filter TEXT  Filter documents by filename pattern
  --no-sources       Hide source citations
```

#### `ask` - Single question
```bash
corpus ask <question> [OPTIONS]

Options:
  -f, --filter TEXT  Filter documents by filename pattern
  --no-sources       Hide source citations
```

### Diagram Generation

#### `diagram` - Generate diagrams
```bash
corpus diagram [query] [OPTIONS]

Options:
  -t, --type TEXT         Diagram type (flowchart, network, etc.)
  --theme TEXT            Color theme (default, professional, vibrant)
  -l, --layout TEXT       Layout engine (dagre, elk)
  -s, --search TEXT       Create from document search
  -e, --export TEXT       Export format (png, pdf, etc.)
```

#### `diagram-batch` - Batch generation
```bash
corpus diagram-batch [OPTIONS]

Options:
  -f, --file PATH    File containing queries (one per line)
  -t, --type TEXT    Default diagram type
  --theme TEXT       Color theme
```

#### `diagram-gallery` - Browse diagrams
```bash
corpus diagram-gallery [OPTIONS]

Options:
  -l, --limit INTEGER    Number of diagrams to show [default: 20]
  -f, --format TEXT      Filter by format (svg, png, pdf)
  -s, --sort TEXT        Sort by: modified, created, name, size
```

### Database Management

#### `status` - Show database statistics
```bash
corpus status
```

#### `list` - List indexed documents
```bash
corpus list [OPTIONS]

Options:
  -s, --sort TEXT     Sort by: name, size, date, chunks
  -f, --filter TEXT   Filter by filename pattern
  -l, --limit INTEGER Number to show [default: 20]
```

#### `clear` - Clear database
```bash
corpus clear [OPTIONS]

Options:
  -y, --yes  Skip confirmation prompt
```

#### `remove` - Remove specific document
```bash
corpus remove <filename>
```

### Utility Commands

#### `analyze` - Preview directory
```bash
corpus analyze <directory>
```

#### `export` - Export conversation
```bash
corpus export [OPTIONS]

Options:
  -o, --output TEXT  Output filename
```

#### `supported` - Show supported file types
```bash
corpus supported
```

#### `info` - System information
```bash
corpus info
```

## Diagram Types 🎨

The diagram generator supports multiple types:

- **flowchart**: Process flows, decision trees, workflows
- **network**: System architecture, infrastructure diagrams
- **hierarchy**: Organizational charts, tree structures
- **sequence**: Interaction diagrams, communication flows
- **erd**: Entity relationship diagrams for databases
- **state**: State machines and transitions
- **mind_map**: Concept maps, brainstorming
- **gantt**: Project timelines (requires D2 Pro)

### Diagram Examples

```bash
# Flowchart with theme
corpus diagram "user registration process" -t flowchart --theme professional

# Network diagram from documents
corpus diagram --search "microservices architecture" -t network

# Batch generation with type hints
echo "flowchart:CI/CD pipeline" > diagrams.txt
echo "network:AWS architecture" >> diagrams.txt
echo "sequence:API authentication flow" >> diagrams.txt
corpus diagram-batch -f diagrams.txt --theme vibrant
```

## Configuration ⚙️

Corpus uses a flexible configuration system that can be managed entirely through the CLI.

### Quick Setup

Run the interactive configuration wizard:

```bash
corpus config setup
```

This will guide you through setting up:
- API keys
- Model selection
- Storage paths
- Processing parameters

### Configuration Commands

#### View Configuration
```bash
# Show all settings
corpus config show

# Show specific setting
corpus config show GEMINI_API_KEY
corpus config show CHUNK_SIZE

# Show full details including lists
corpus config show --all
```

#### Modify Settings
```bash
# Set individual values
corpus config set GEMINI_API_KEY "your-api-key"
corpus config set GEMINI_MODEL "gemini-1.5-pro"
corpus config set CHUNK_SIZE 1500
corpus config set MAX_FILE_SIZE_MB 200

# Set lists (use JSON format)
corpus config set CODE_EXTENSIONS '["py", "js", "ts", "java"]'
```

#### Reset Configuration
```bash
# Reset everything to defaults
corpus config reset

# Reset specific setting
corpus config reset CHUNK_SIZE
```

#### Validate Configuration
```bash
# Check for any issues
corpus config validate
```

#### Export Configuration
```bash
# Export to different formats
corpus config export --format json > my-config.json
corpus config export --format yaml > my-config.yaml
corpus config export --format env > .env.example
```

### Configuration Options

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| **API Settings** |
| GEMINI_API_KEY | string | - | Your Google Gemini API key |
| GEMINI_MODEL | string | gemini-1.5-flash | Model to use for chat |
| GEMINI_IMG_MODEL | string | gemini-pro-vision | Model for image analysis |
| **Storage** |
| DB_PATH | string | ~/.corpus/research.db | ChromaDB database location |
| DIAGRAMS_PATH | string | ~/.corpus/diagrams | Where to save generated diagrams |
| COLLECTION_NAME | string | documents | ChromaDB collection name |
| **Processing** |
| CHUNK_SIZE | int | 1000 | Text chunk size for indexing |
| CHUNK_OVERLAP | int | 200 | Overlap between chunks |
| MIN_CHUNK_LENGTH | int | 50 | Minimum chunk size |
| MAX_FILE_SIZE_MB | int | 100 | Maximum file size to index |
| **Search** |
| MAX_RESULTS | int | 5 | Default search results |
| MAX_MEMORY | int | 10 | Chat memory length |
| **File Types** |
| CODE_EXTENSIONS | list | [many] | Code file extensions to index |
| IMAGE_EXTENSIONS | list | [.jpg, .png, etc] | Image formats to process |
| TEXT_EXTENSIONS | list | [.txt, .md, .pdf] | Text document formats |

### Environment Variables

You can also use environment variables, especially for sensitive data:

```bash
# .env file
GEMINI_API_KEY=your-api-key-here

# Or export directly
export GEMINI_API_KEY="your-api-key"
```

The configuration system checks environment variables first for API keys.

### Configuration Location

Configuration is stored in:
- Config file: `~/.corpus/config.json`
- Env file: `.env` (in current directory)

To see the exact location:
```bash
corpus config path
```

## Advanced Usage 🔧

### Configuration Management

```bash
# Change models on the fly
corpus config set GEMINI_MODEL gemini-1.5-pro

# Increase chunk size for larger documents
corpus config set CHUNK_SIZE 2000

# Add custom file extensions
corpus config set CODE_EXTENSIONS '["py","js","rs","go","java"]'

# Validate settings
corpus config validate
```

### Multi-threaded Indexing
```bash
# Auto-detect optimal threads
corpus index ~/LargeCollection

# Specify thread count
corpus index ~/LargeCollection --workers 16

# Single-threaded mode (for debugging)
research index ~/LargeCollection --no-threading
```

### Document Filtering
```bash
# Chat with specific documents
corpus chat --filter "machine_learning*.pdf"

# Ask about specific topics
corpus ask "explain transformers" --filter "deep_learning/*"
```

### Diagram Workflows
```bash
# Generate, then export to different format
corpus diagram "database schema" --type erd
corpus diagram-gallery  # Find the generated file
corpus diagram "query_diagram_20240115_143022.svg" --export png
```

## Tips & Tricks 💡

1. **Indexing Performance**: Use `--workers` to optimize for your system. Generally, 2-4x CPU cores works well.

2. **Search Quality**: Index related documents together for better context in answers.

3. **Diagram Generation**: 
   - Be specific in descriptions for better results
   - Use document search for accuracy when creating architecture diagrams
   - Try different themes for various presentation contexts

4. **Memory Usage**: For large collections, index in batches using `--pattern`.

5. **GitHub Indexing**: Use a personal access token for better rate limits and private repo access.

## Troubleshooting 🔧

### Common Issues

1. **"D2 renderer not found"**
   - Install D2: `curl -fsSL https://d2lang.com/install.sh | sh -s --`
   - Ensure D2 is in your PATH

2. **"API key not found"**
   - Run `corpus config setup` to configure
   - Or set directly: `corpus config set GEMINI_API_KEY "your-key"`
   - Check with: `corpus config show GEMINI_API_KEY`

3. **Configuration issues**
   - Validate config: `corpus config validate`
   - Reset to defaults: `corpus config reset`
   - Check location: `corpus config path`

4. **Indexing failures**
   - Check file permissions
   - Ensure files aren't corrupted
   - Try with `--no-threading` for debugging
   - Check max file size: `corpus config show MAX_FILE_SIZE_MB`

5. **Out of memory**
   - Reduce chunk size: `corpus config set CHUNK_SIZE 500`
   - Index in smaller batches
   - Use `--workers 1` to reduce memory usage

## Contributing 🤝

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License 📄

MIT License - see LICENSE file for details

## Acknowledgments 🙏

- Built with [ChromaDB](https://www.trychroma.com/) for vector storage
- Powered by [Google Gemini](https://ai.google.dev/) for AI capabilities
- Diagrams rendered with [D2](https://d2lang.com/)
- CLI interface using [Typer](https://typer.tiangolo.com/)
- Beautiful formatting with [Rich](https://rich.readthedocs.io/)