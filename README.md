# CorpusNote CLI

## Chat with Your Documents, Powered by AI

This project is a standalone Command-Line Interface (CLI) application designed to serve as a **Personal Research Assistant**. Its core function is to empower users by allowing them to ingest their own diverse collection of documents—including PDFs, plain text files, Markdown files, and various image formats like JPG and PNG—into a robust, local, and searchable knowledge base. Once indexed, you can interact with this rich information using a powerful Artificial Intelligence, ensuring that the answers you receive are directly sourced and verifiable from your specific data.


![Example for Daedaelus](/response.png)
---

##  Features

* **Intelligent Document Ingestion:**
    * **Flexible Indexing:** Easily index individual files or recursively process entire directories containing your research materials.
    * **Smart Chunking:** Documents are not merely stored; they are intelligently split into optimized, manageable "chunks." This process, often based on sentence boundaries and incorporating context-preserving overlap, is crucial for efficient retrieval by the AI. Each chunk is designed to be self-contained yet connected to its surrounding information.
    * **Robust Text Cleaning:** Addresses common challenges in document processing, such as fixing hyphenated words split across lines, removing problematic characters (like null bytes often found in PDFs), and normalizing inconsistent whitespace. This ensures high-quality data for the AI to process.
    * **Deduplication & Efficient Updates:** Leverages file hashing to uniquely identify documents. This prevents redundant indexing of unchanged files and allows for swift updates, where only new or modified documents are processed, saving time and resources.
* **Context-Aware Conversational AI:**
    * **Natural Language Interaction:** Ask questions in plain English, and the AI will delve into your indexed documents for answers.
    * **Retrieval-Augmented Generation (RAG):** This is the core intelligence. When you ask a question, the system dynamically retrieves the most semantically relevant document chunks from your local knowledge base. These retrieved snippets then *augment* the prompt sent to the Large Language Model (LLM). This grounding mechanism significantly minimizes AI "hallucinations," ensuring factual accuracy directly from your provided data.
    * **Google Gemini Integration:** The application seamlessly integrates with powerful Google Gemini models (specifically those with multimodal capabilities like Gemini 1.5 Flash), utilizing their advanced understanding and generation abilities to formulate insightful and coherent answers.
    * **Multimodal Understanding:** Beyond just text, the system can process and understand information from images. It extracts textual content via OCR and generates detailed semantic descriptions of visual elements using Gemini's vision capabilities, incorporating this information into the searchable knowledge base.
    * **Conversation Memory:** Maintains a short-term memory of your ongoing chat history. This conversational context is passed to the LLM, enabling more natural follow-up questions and coherent multi-turn dialogues.
    * **Transparent Source Citations:** Every answer is backed by clear citations, detailing the specific document, page number, and even the exact chunk from which the information was derived. For image-based answers, it includes image dimensions and format, enhancing transparency and trustworthiness.
* **Comprehensive Document Management:**
    * **Powerful CLI Commands:** A rich set of command-line interface tools provides full control over your knowledge base:
        * `index`: Add new files or entire folders.
        * `update`: Efficiently refresh the index with only new or changed files.
        * `watch`: Monitor a directory in the background for continuous auto-indexing.
        * `list`: Display all indexed documents with sorting and filtering options.
        * `status`: Get a detailed overview of your database, including content type breakdowns.
        * `analyze`: Preview what would be indexed from a directory without committing changes.
        * `remove`: Delete specific documents from your index.
        * `clear`: Empty your entire knowledge base.
        * `export`: Save your chat conversation history to a Markdown file.
* **User-Friendly Command-Line Interface:**
    * Developed using `typer` for defining robust, intuitive command-line arguments and `rich` for rendering beautiful, informative, and colored terminal output. This includes interactive prompts, clear progress bars during indexing, well-formatted tables for lists and statistics, and visually appealing panels for key information.

---

## ⚙️ Technologies Used

* **Python:** The core programming language orchestrating all components.
* **Google Gemini API:** Provides the cutting-edge Large Language Model and multimodal (vision) capabilities for understanding and generating text/insights.
* **ChromaDB:** A lightweight, embeddable, and persistent vector database, specifically chosen for its efficiency in storing and performing semantic similarity searches on document embeddings locally.
* **PyPDF:** Utilized for robust and reliable text extraction from PDF documents.
* **Pillow (PIL Fork):** Essential for general image processing, including resizing, format conversions, and preparing images for OCR and Gemini Vision.
* **PyMuPDF (Fitz):** Recommended for more advanced and robust image extraction from PDFs, complementing Pillow.
* **`opencv-python` (cv2):** Used for image preprocessing techniques (like denoising, thresholding) to enhance images for better OCR accuracy.
* **`pytesseract` & `easyocr`:** Multiple OCR (Optical Character Recognition) libraries are employed as robust fallback mechanisms to extract text from images and diagrams.
* **Typer:** A modern library used to build the intuitive and type-hinted command-line interface, providing a smooth user experience.
* **Rich:** A powerful library for terminal rendering, responsible for the beautiful and interactive console output, including colored text, panels, tables, and animated progress bars.

---

## Getting Started

Follow these steps to set up and run the Research Assistant CLI on your local machine.

### Prerequisites

* **Python 3.9+**: Ensure you have a compatible Python version installed.
* **Google Gemini API Key**: Obtain your API key from [Google AI Studio](https://aistudio.google.com/app/apikey). This key is essential for interacting with the Gemini models.

### Installation

1.  **Clone the repository:**
    Begin by cloning the project repository from GitHub to your local machine.
    ```bash
    git clone [https://github.com/your-username/CorpusNote.git](https://github.com/your-username/CorpusNote.git)
    cd research-assistant-cli
    ```
    
2.  **Create and activate a virtual environment (highly recommended):**
    Using a virtual environment isolates your project's dependencies, preventing conflicts with other Python projects.
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On macOS/Linux
    .\venv\Scripts\activate   # On Windows
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file in the root of your project directory with the following list of libraries. This ensures all necessary packages are installed.
    ```
    google-generativeai
    chromadb
    pypdf
    typer
    rich
    python-dotenv
    Pillow
    PyMuPDF
    opencv-python # For image preprocessing
    pytesseract   # For OCR
    easyocr       # For alternative OCR
    ```
    Then, install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For `pytesseract`, you might also need to install the Tesseract OCR engine executable on your system. Refer to the `pytesseract` [documentation](https://pypi.org/project/pytesseract/) for platform-specific instructions.*

4.  **Configure your API Key:**
    Your Gemini API key needs to be securely stored and accessible to the application. Create a file named `.env` in the root of your project directory:
    ```
    # .env
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```
    **Remember to replace `"YOUR_GEMINI_API_KEY"` with the actual API key you obtained from Google AI Studio.**

    Ensure your `config.py` file is set up to load this environment variable and specifies the correct Gemini models for multimodal support:
    ```python
    # config.py
    import os
    from dotenv import load_dotenv

    load_dotenv() # This loads environment variables from the .env file

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Set it as an environment variable or in a .env file.")

    # Use a multimodal model for both chat responses and image analysis
    GEMINI_MODEL = "gemini-1.5-flash"
    GEMINI_IMG_MODEL = "gemini-1.5-flash"

    DB_PATH = "./chroma_db"          # Path where your ChromaDB data will be stored
    COLLECTION_NAME = "research_documents" # Name of your document collection in ChromaDB

    MAX_FILE_SIZE_MB = 100           # Maximum size of a file (in MB) to be processed
    CHUNK_SIZE = 500                 # Target number of words per text chunk
    CHUNK_OVERLAP = 50               # Number of words to overlap between consecutive text chunks
    MIN_CHUNK_LENGTH = 100           # Minimum number of characters for a valid text chunk

    SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"] # All supported file types
    MAX_RESULTS = 5                  # Number of top relevant chunks to retrieve from ChromaDB for context
    ```

### Basic Usage

All interactions are done via the `main.py` script. Run it as `python main.py <command> [options]`. When processing images, the system will now extract any legible text via OCR and generate detailed textual descriptions of the visual content using Gemini's vision capabilities. These descriptions are then indexed alongside your text documents.

#### Key Commands:

1.  **`index <path>`**: Ingest documents (files or directories) into your knowledge base.
    * **Index a single PDF file:**
        ```bash
        python main.py index /path/to/your/document.pdf
        ```
    * **Index a standalone image file:**
        ```bash
        python main.py index /path/to/your/diagram.jpg
        ```
    * **Index an entire directory recursively:**
        ```bash
        python main.py index ~/MyResearchPapers --recursive
        ```
    * **Force re-indexing of existing files in a directory:**
        ```bash
        python main.py index ./project_docs --force
        ```
    * **Index only specific files matching a pattern:**
        ```bash
        python main.py index /path/to/notes --pattern "*meeting_notes*.md"
        ```

2.  **`chat`**: Starts an interactive AI chat session with your indexed documents.
    ```bash
    python main.py chat
    ```
    * **Commands within the chat session:**
        * `clear`: Clears the current conversation history.
        * `sources`: Toggles the display of source citations for AI responses.
        * `stats`: Shows a summary of the indexed database.
        * `filter images`: Temporarily filters search results to only image-derived content.
        * `filter text`: Temporarily filters search results to only text-based document content.
        * `clear filter`: Removes any active content type filter.
        * `export`: Exports the current chat conversation to a Markdown file.
        * `exit` / `quit` / `q`: Exits the chat session.

3.  **`ask <question>`**: Ask a single question about your documents in non-interactive mode.
    ```bash
    python main.py ask "What are the key findings discussed in the introduction of the report?"
    python main.py ask "Describe the process illustrated in the flowchart." --filter "process_diagram.png"
    python main.py ask "What objects are visible in the image of the forest?" --filter "forest_walk.jpg"
    python main.py ask "Summarize the conclusion without citing sources." --no-sources
    ```

4.  **`status`**: Shows an overview and detailed statistics of your ChromaDB knowledge base.
    ```bash
    python main.py status
    ```

5.  **`list`**: Lists all indexed documents with options for sorting and filtering.
    ```bash
    python main.py list
    python main.py list --sort size --limit 10 # List top 10 largest documents
    python main.py list --filter "report"      # List documents with "report" in their name
    ```

6.  **`update <path>`**: Efficiently updates the index by processing only new or modified files within a specified directory.
    ```bash
    python main.py update ~/MyResearchPapers
    ```

7.  **`watch <path>`**: Monitors a directory for changes and automatically indexes new or modified files at a set interval.
    ```bash
    python main.py watch ~/ObservatoryData --interval 300 # Checks every 5 minutes
    ```

8.  **`analyze <path>`**: Provides an analysis of a directory's contents without performing actual indexing. Useful for planning.
    ```bash
    python main.py analyze ~/ProspectiveDocuments
    ```

9.  **`remove <filename>`**: Deletes a specific document (and all its associated chunks) from the index.
    ```bash
    python main.py remove "outdated_article.pdf"
    ```

10. **`clear`**: Erases all indexed documents and chunks from the entire database.
    ```bash
    python main.py clear # Requires confirmation
    python main.py clear --yes # Skips confirmation
    ```

11. **`export`**: Exports the current chat conversation history to a Markdown file.
    ```bash
    python main.py export # Exports to a timestamped file
    python main.py export --output "my_project_discussion.md" # Exports to a specific file
    ```

12. **`supported`**: Lists all file types currently supported for indexing.
    ```bash
    python main.py supported
    ```

---
