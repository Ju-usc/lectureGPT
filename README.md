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



## Notes

- Supports PDF and Jupyter notebook formats
- Handles text and code content
- Skips image-containing cells
- Uses GPT-3.5-turbo and GPT-4 for content enhancement
- Generates embeddings for vector search

## Requirements
```
openai>=1.0.0
Pillow>=10.0.0
PyPDF2>=3.0.0
pdf2image>=1.16.3
python-dotenv>=1.0.0
```

## Environment Setup
1. Create a `.env` file in the root directory
2. Add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```