# Research Assistant CLI

## Chat with Your Documents, Powered by AI

This project is a standalone Command-Line Interface (CLI) application that acts as a **Personal Research Assistant**. It enables you to ingest your own documents (PDFs, TXT, Markdown, JPG, PNG) into a local, searchable knowledge base and then interact with that knowledge using a powerful AI, ensuring answers are directly sourced from your data.

**Conceived and built as a one-day internship project with the assistance of Claude**, this tool demonstrates rapid prototyping and the significant acceleration offered by modern AI development.

---

## ✨ Features

* **Intelligent Document Ingestion:**
    * Index individual files or entire directories.
    * **Smart Chunking:** Documents are split into optimized, context-preserving chunks for efficient retrieval.
    * **Text Cleaning:** Addresses common extraction issues for improved data quality.
    * **Deduplication & Updates:** Uses file hashes to prevent re-indexing and efficiently updates modified documents.
* **Context-Aware Chat:**
    * Ask natural language questions about your indexed content.
    * **Retrieval-Augmented Generation (RAG):** The system retrieves relevant document chunks to inform the LLM's response, minimizing "hallucinations" and ensuring factual accuracy.
    * **Google Gemini Integration:** Utilizes Google Gemini models for generating insightful and coherent answers.
    * **Conversation Memory:** Maintains short-term chat history for natural follow-up questions.
    * **Source Citations:** Clearly cites documents, pages, and chunks, enhancing transparency.
* **Comprehensive Document Management:**
    * Commands to index, update, watch directories, list documents, check status, analyze potential indexing, remove specific documents, clear the database, and export chat conversations.
* **User-Friendly CLI:** Built with `typer` for robust commands and `rich` for visually appealing and informative terminal output.

---

## ⚙️ Technologies Used

* **Python:** Core programming language.
* **Google Gemini API:** For LLM capabilities.
* **ChromaDB:** Lightweight, embeddable vector database for semantic search.
* **PyPDF:** For robust PDF text extraction.
* **Typer:** For building the command-line interface.
* **Rich:** For enhanced terminal output.
* **Pillow, PyMuPDF:** For image processing and advanced PDF handling.

---

## 🚀 Getting Started

Follow these steps to set up and run the Research Assistant CLI.

### Prerequisites

* Python 3.9+
* A Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/research-assistant-cli.git](https://github.com/your-username/research-assistant-cli.git)
    cd research-assistant-cli
    ```
    *Replace `your-username/research-assistant-cli.git` with the actual repository URL.*

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` with:
    ```
    google-generativeai
    chromadb
    pypdf
    typer
    rich
    python-dotenv
    Pillow
    PyMuPDF
    ```
    Then run: `pip install -r requirements.txt`

4.  **Configure your API Key:**
    Create a `.env` file in the project root:
    ```
    # .env
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```
    **Replace `YOUR_GEMINI_API_KEY` with your actual key.**

    Ensure `config.py` loads this key:
    ```python
    # config.py
    import os
    from dotenv import load_dotenv

    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Set it as an environment variable or in a .env file.")

    GEMINI_MODEL = "gemini-1.5-flash"
    DB_PATH = "./chroma_db"
    COLLECTION_NAME = "research_documents"
    MAX_FILE_SIZE_MB = 100
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    MIN_CHUNK_LENGTH = 100
    SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".jpg", ".jpeg", ".png"]
    MAX_RESULTS = 5
    ```

### Basic Usage

Run `python main.py <command> [options]`. Images from PDFs and standalone files (`.jpg`, `.jpeg`, `.png`) will be processed via Gemini for textual descriptions, which are then indexed.

#### 1. Index Your Documents

**Index file/directory:**
```bash
python main.py index /path/to/your/document.pdf
python main.py index /path/to/your/image.jpg
python main.py index /path/to/your/research_papers --recursive
python main.py index ~/Documents/Reports --force
python main.py index /path/to/my/docs --pattern "*atomic*.pdf"
```


#### 2. Start a Chat Session
AI can now understand questions requiring both text and image context.

Interactive chat:
```shell
python main.py chat
```

Commands: `clear`, `sources`, `stats`, `exit`.

#### 3. Ask a single question:
```shell
python main.py ask "What is the main challenge in implementing RAG systems?"
python main.py ask "Describe the diagram on page 3." --filter "my_report.pdf"
python main.py ask "What is in the image of the cat?" --filter "cat_picture.jpg"
python main.py ask "Summarize the conclusion." --no-sources
```

#### 4. Manage your local database
```shell
python main.py status
python main.py list [--sort size|date|chunks|name] [--filter "pattern"] [--limit N]
python main.py update /path/to/your/research_papers
python main.py watch /path/to/your/docs --interval 300
python main.py analyze /path/to/your/data
python main.py remove "old_notes.md"
python main.py clear [--yes]
```

#### 5. View Supported File Types
```bash
python main.py supported
```
