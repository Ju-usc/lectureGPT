# LectureGPT

A Retrieval-Augmented Generation (RAG) system for processing and querying lecture materials.

## Overview

This system processes lecture materials (PDFs and Jupyter notebooks) to create an AI-powered question-answering system. It extracts, enhances, and embeds content for efficient retrieval.

## Installation

```bash
pip install -r requirements.txt
```

## Functions Documentation

### Content Processing

#### `codeLLM(text: str) -> str`
- Processes code and markdown content from Jupyter notebooks
- Enhances content for RAG retrieval
- Uses GPT-3.5-turbo with specialized prompts
- Returns enhanced, searchable text or 'NONE' if content is irrelevant

#### `slideLLM(text: str) -> str`
- Processes lecture slide content from PDFs
- Expands abbreviations and bullet points
- Uses GPT-4 with academic-focused prompts
- Returns enhanced, searchable text or 'NONE' if content is irrelevant

### Document Processing

#### `pdf2content(file_path: str) -> List[Dict[str, Any]]`
- Extracts content from PDF files
- Includes metadata (title, author, etc.)
- Returns list of dictionaries with page content and metadata
- Each dictionary includes file_format='pdf'

#### `colab2content(file_path: str) -> List[Dict[str, Any]]`
- Processes Jupyter/Colab notebooks
- Skips image-containing cells
- Returns list of dictionaries with cell content
- Each dictionary includes file_format='ipynb'

#### `file2content(file_paths: List[str]) -> List[Dict[str, Any]]`
- Main document processing function
- Handles both PDF and notebook files
- Routes to appropriate processing function
- Returns combined list of processed content

### Content Enhancement

#### `get_enhance_content(content_list: List[Dict[str, Any]]) -> List[str]`
- Routes content to appropriate LLM (codeLLM or slideLLM)
- Processes based on file_format
- Returns list of enhanced content strings

### Content Chunking and Embedding

#### `chunk_content(enhanced_content: List[str], window_size: int, overlap: int = 1) -> List[str]`
- Splits content into overlapping chunks
- Uses sliding window approach
- Parameters:
  * window_size: Number of items per chunk
  * overlap: Number of overlapping items

#### `embed_text(text: str) -> np.ndarray`
- Generates embeddings using OpenAI's text-embedding-ada-002
- Returns numpy array of embedding vector

#### `chunks2embeddings(chunks: List[str]) -> List[np.ndarray]`
- Converts text chunks to embeddings
- Returns list of embedding vectors

## System Architecture

### 1. Basic RAG System

#### Data Processing Pipeline
- **Input**: Lecture slides (.pdf) and code demos (.ipynb)
- **Output**: Generated answers based on user queries
- **Processing Flow**:
  1. Content Extraction & Preparation
  2. Embedding Generation
  3. Vector Database Storage
  4. Query Processing
  5. Answer Generation

### Vector Store Management

#### `VectorStore`
- Manages document storage and retrieval
- Handles saving/loading of vector database
- Supports direct text updates and file-based updates
- Key methods:
  * `add_documents`: Add new documents to store
  * `save`: Save vector store to disk
  * `load`: Load vector store from disk
  * `get_stored_files`: List stored documents

### QA Generation

#### `QAGenerator`
- Main interface for question answering
- Manages conversation history and context
- Supports knowledge base updates
- Key methods:
  * `generate_answer`: Generate answers from questions
  * `add_documents`: Add documents to knowledge base
  * `add_text_to_knowledge`: Directly add text content
  * `generate_knowledge_suggestion`: Generate suggestions for missing knowledge

### Web Interface

#### Streamlit App
- Interactive web interface for Q&A
- Student and instructor views
- Features:
  * Real-time question answering
  * Knowledge base management
  * Content suggestions for unanswered questions
  * Interactive knowledge updates
  * System prompt configuration

## Usage

### Starting the App
```bash
streamlit run streamlit_app.py
```

### Student View
- Ask questions about lecture content
- View source references for answers
- Track conversation history

### Instructor View
1. Knowledge Base Management
   - Upload lecture materials (PDF/Jupyter)
   - View stored documents
   - Monitor knowledge base status

2. Q&A Monitoring
   - Review student questions
   - See answer sources
   - Get suggestions for missing knowledge

3. Knowledge Updates
   - Add missing information
   - Update system responses
   - Configure QA behavior

## System Components

### 1. Content Processing (`parsing.py`)
- Extracts content from files
- Enhances content using LLMs
- Prepares text for vectorization

### 2. Vector Store (`vector_store.py`)
- Manages document embeddings
- Handles persistence
- Provides similarity search

### 3. Retriever (`retriever.py`)
- Implements retrieval logic
- Ranks and filters results
- Manages context window

### 4. QA Generator (`generator.py`)
- Orchestrates QA process
- Manages conversation
- Handles knowledge updates

### 5. Web Interface (`streamlit_app.py`)
- Provides user interface
- Manages session state
- Handles file uploads

## Configuration

### Environment Variables
Required in `.env`:
```
OPENAI_API_KEY=your_api_key
```

### System Settings
- Vector store path: "lecture_db" (default)
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Temperature: 0 (for consistent answers)

## Dependencies
```
streamlit
openai>=1.0.0
langchain
faiss-cpu
python-dotenv
pypdf2
python-pptx
numpy
```

## Notes

- Supports PDF and Jupyter notebook formats
- Handles text and code content
- Skips image-containing cells
- Uses GPT-3.5-turbo and GPT-4 for content enhancement
- Generates embeddings for vector search
- Uses OpenAI's text-embedding-ada-002 for embeddings
- Supports incremental knowledge base updates
- Maintains conversation history
- Provides source references for answers