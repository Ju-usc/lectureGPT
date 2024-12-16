import os
from typing import Dict, List, Any, Tuple
import base64
import tempfile
from PyPDF2 import PdfReader
from pptx import Presentation
from openai import OpenAI
from dotenv import load_dotenv
import itertools
import json
import re
import numpy as np
import concurrent.futures
# Initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def codeLLM(text: str) -> str:

    """Process raw code content to enhance it for RAG retrieval.

    Args:
        text: Raw code content to process

    Returns:
        str: Enhanced code content optimized for RAG retrieval
    """
    try:
        SYSTEM_PROMPT = """You are an expert processor specializing in transforming code and markdown content from Jupyter notebooks into detailed, explanatory text optimized for RAG (Retrieval Augmented Generation) systems.

        Your task is to transform code and markdown cells into detailed, well-structured explanations that:

        1. Preserve complete technical accuracy and terminology.
        2. Explain the purpose and functionality of the code in clear, concise language.
        3. Expand any comments or variable names into full explanations where appropriate.
        4. Add implicit context that would help in retrieval.
        5. For markdown cells, expand abbreviations and clarify any shorthand notations.
        6. Format content into coherent paragraphs.
        
        Important Guidelines:

        - Focus on enriching the content while staying true to the original code's functionality and intent.
        - Do not introduce information not implied by the original content.
        - Keep the enhanced content relevant and concise.
        - Format output as clear, searchable text.
        - Do not be verbose.
        - Do not include any meta-commentary or acknowledgments.
        - If the content is too simple or not relevant, respond with NONE.
        - If the content is too short to be retrieved as RAG, respond with NONE.
        """

        CONTENT_PROMPT = f"""
        <content>
        {text}
        </content>

        Transform this notebook content into an enhanced version optimized for RAG retrieval.

        Respond only with the enhanced content. If the content is not relevant or too short, respond with NONE.
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": CONTENT_PROMPT
                }
            ],
            max_tokens=300,
            temperature=0.4
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error processing content: {str(e)}")
        return text

def slideLLM(text: str) -> str:
    """Process raw content from lecture slides to enhance it for RAG retrieval.
    
    Args:
        text: Raw text content to process
        
    Returns:
        str: Enhanced content optimized for RAG retrieval
    """
    try:
        SYSTEM_PROMPT = """You are an expert academic content processor specialized in enhancing raw lecture slide content for RAG (Retrieval Augmented Generation) systems.

        Your task is to transform raw lecture slide content into detailed, well-structured explanations that:
        1. Preserve complete technical accuracy and terminology
        2. Expand bullet points and abbreviations into full, clear explanations
        3. Add implicit context that would help in retrieval
        4. Maintain academic tone and precision
        5. Format content into coherent paragraphs

        Important Guidelines:
        - Focus on enriching the raw content while staying true to the original meaning
        - Never introduce information not implied by the original content
        - Keep the enhanced content relevant and concise
        - Format output as clear, searchable text
        - Do not be verbose
        - Do not include any meta-commentary or acknowledgments
        - Some raw content is too simple to enhance or irrelevant, in which case, respond with NONE
        - Some raw content is too short to be retrieved as RAG, in which case, response with NONE
        """

        CONTENT_PROMPT = f"""
        <content>
        {text}
        </content>

        Transform this lecture content into an enhanced version optimized for RAG retrieval.
        Respond only with the enhanced content. If the content is not relevant or too short, respond with NONE.
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": CONTENT_PROMPT
                }
            ],
            max_tokens=300,
            temperature=0.4, 
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error processing content: {str(e)}")
        return text

def pdf2content(file_path: str) -> List[Dict[str, Any]]:
    """Process a PDF file and extract content and metadata from each page.
    
    Extracts both text content and various metadata including:
    - Raw text, page number, total pages
    - Document properties (title, author, subject, etc.)
    """
    try:
        content = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            # Extract document-level metadata
            doc_info = pdf_reader.metadata
            doc_metadata = {
                "file_path": file_path,
                "title": doc_info.get("/Title", ""),
                "author": doc_info.get("/Author", ""),
                "subject": doc_info.get("/Subject", ""),
                "creator": doc_info.get("/Creator", ""),
                "producer": doc_info.get("/Producer", ""),
                "creation_date": doc_info.get("/CreationDate", ""),
                "modification_date": doc_info.get("/ModDate", ""),
            }
            
            for idx, page in enumerate(pdf_reader.pages, 1):
                # Extract text
                raw_text = page.extract_text()
                
                content.append({
                    'file_format': 'pdf',
                    "page_number": idx,
                    "total_pages": total_pages,
                    "raw_text": raw_text,
                    "doc_metadata": doc_metadata
                })
        return content
        
    except Exception as e:
        print(f"Error processing PDF {file_path}: {str(e)}")
        return []

def colab2content(file_path: str) -> List[Dict[str, Any]]:
    """Process a Jupyter/Colab notebook file and extract content from each cell without AI processing, skipping image cells."""
    try:
        
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        content = []
        total_cells = len(notebook['cells'])
        
        for index, cell in enumerate(notebook['cells'], start=1):
            cell_type = cell['cell_type']
            
            raw_text = ''.join(cell.get('source', []))
            # Skip cells with images
            if 'data:image' in raw_text:
                continue

            content.append({
                'file_path': file_path,
                'file_format': 'ipynb',
                'cell_type': cell_type,
                'raw_text': raw_text,
                'cell_number': index,
                'total_cells': total_cells
            })
        
        return content
    except Exception as e:
        print(f"Error processing notebook: {str(e)}")
        return []

def file2content(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Extract content from multiple documents without AI processing"""
    print(f"Processing files: {file_paths}")
    
    all_content = []
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        if file_path.endswith('.pdf'):
            content = pdf2content(file_path)
            print(f"PDF content length: {len(content)}")
        elif file_path.endswith('.ipynb'):
            content = colab2content(file_path)
            print(f"Notebook content length: {len(content)}")
        else:
            print(f"Unsupported file type: {file_path}")
            continue
            
        all_content.extend(content)
    
    print(f"Total content items: {len(all_content)}")
    return all_content

def get_enhance_content(content_list: List[Dict[str, Any]]) -> List[str]:
    """Process content using appropriate LLM based on file format.
    
    Args:
        content_list: List of dictionaries containing content and metadata
        
    Returns:
        List of enhanced content strings
    """
    if not content_list:
        return []

    def process_single_item(content_dict: Dict[str, Any]) -> str:
        content = content_dict.get('raw_text', '')
        file_format = content_dict.get('file_format', '')
        
        if not content:
            return ''
            
        if file_format == 'ipynb':
            return codeLLM(content)
        else:  # default to pdf
            return slideLLM(content)

    # Process items in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(content_list), 5)) as executor:
        enhanced_content = list(executor.map(process_single_item, content_list))
    
    return enhanced_content

def chunk_content(enhanced_content: List[str], window_size: int, overlap: int = 1) -> List[str]:
    """Split enhanced content into overlapping chunks using sliding window.
    
    Args:
        enhanced_content: List of enhanced text content
        window_size: Number of items in each chunk
        overlap: Number of items to overlap between chunks
    
    Returns:
        List of chunks, where each chunk contains window_size items (except possibly the last chunk)
    """
    if not enhanced_content:
        return []
    
    # Filter out 'NONE' responses and empty strings

    chunks = []
    stride = window_size - overlap
    
    # Create chunks with sliding window
    for i in range(0, len(enhanced_content), stride):
        chunk = enhanced_content[i:i + window_size]
        if len(chunk) >= overlap:  # Only add chunks that have at least 'overlap' items
            chunks.append(' '.join(chunk))
    
    return chunks

def embed_text(text: str) -> np.ndarray:
    """Generate embeddings for a given text using OpenAI's text-embedding-ada-002"""
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return np.array(response.data[0].embedding)

def chuncks2embeddings(chunks: List[str]) -> List[np.ndarray]:
    """Generate embeddings for a list of text chunks"""
    return [embed_text(chunk) for chunk in chunks]

def process_documents(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Main function to process multiple documents"""
    # Step 1: Extract content from all documents
    print("\nExtracting content from documents...")
    extracted_content = file2content(file_paths)

    print("Successfully Extracted content")
    for item in extracted_content:
        print(f"File Format: {item['file_format']}")
        print(f"Raw Text: {item['raw_text']}")
        print("\n")


    # Step 2: Enhance content using GPT-4
    print("\nEnhancing content using GPT-4...")
    enhanced_content = get_enhance_content(extracted_content)

    print("Successfully Enhanced content")

    for i in range(len(extracted_content)):
        print(f"Raw Text: {extracted_content[i]['raw_text']}")
        print("\n")
        print(f"Enhanced Text: {enhanced_content[i]}")
        print("\n")

    print("length of enhanced content:", len(enhanced_content))
    print("length of extracted content:", len(extracted_content))
    print("\n")

    for text in enhanced_content:
        if text == 'NONE':
            # get index of unnecessary content
            index = enhanced_content.index(text)
            # remove unnecessary content
            del extracted_content[index]
            del enhanced_content[index]

    print("After removing unnecessary content:\n")
    print("length of enhanced content:", len(enhanced_content))
    print("length of extracted content:", len(extracted_content))

    for i in range(len(extracted_content)):
        print(f"Raw Text: {extracted_content[i]['raw_text']}")
        print("\n")
        print(f"Enhanced Text: {enhanced_content[i]}")
        print("\n")


    # Step 3: get chunks from enhanced content
    print("\nGenerating chunks from enhanced content...")
    chunks = chunk_content(enhanced_content, 2, 1)
    print("Successfully Generated chunks")

    for chunk in chunks:
        print(chunk)
        print("\n")
    

    # Step 4: get embeddings from chunks
    print("\nGenerating embeddings from chunks...")
    embeddings = chuncks2embeddings(chunks)
    print("Successfully Generated embeddings")

    # Perform similarity search
    print("\nPerforming similarity search...")
    query = "What is a retrieval augmented generation?"
    query_embedding = embed_text(query)
    similarity_scores = np.dot(embeddings, query_embedding.T)

    most_similar_index = np.argmax(similarity_scores)
    most_similar_chunk = chunks[most_similar_index]
    most_similar_doc = extracted_content[most_similar_index]

    print(f"Most similar chunk: {most_similar_chunk}")
    print(f"Most similar document: {most_similar_doc}")

    return None


# Process multiple documents at once
if __name__ == "__main__":
    # file_paths = ['./tests/Lecture 13.pdf']
    file_paths = ['./tests/Lab_14_Demo_3_Retrieval_Augmented_Generation_(RAG).ipynb']
    processed_content = process_documents(file_paths)
