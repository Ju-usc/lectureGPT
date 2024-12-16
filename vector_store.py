from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class VectorStore:
    def __init__(self, db_path: str, embedding_model: str = "text-embedding-ada-002"):
        """Initialize vector store with LangChain components.
        
        Args:
            db_path: Path to save/load vector store
            embedding_model: Name of OpenAI embedding model to use
        """
        self.db_path = db_path
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to vector store.
        
        Args:
            documents: List of documents with 'content' and metadata
        """
        print("\n=== Adding Documents to Vector Store ===")
        print(f"Processing {len(documents)} documents...")
        
        # Convert to LangChain Document format
        langchain_docs = []
        for i, doc in enumerate(documents, 1):
            print(f"\nProcessing document {i}/{len(documents)}:")
            content = doc.get('content', '').strip()
            if not content:
                print("- Skipping: Empty content")
                continue
                
            metadata = {
                'source': doc.get('source', 'Unknown'),
                'page': doc.get('page', 1)
            }
            print(f"- Source: {metadata['source']}")
            print(f"- Page: {metadata['page']}")
            print(f"- Content length: {len(content)} characters")
            langchain_docs.append(Document(page_content=content, metadata=metadata))
            
        print(f"\nConverted {len(langchain_docs)} documents to LangChain format")
            
        # Split documents into chunks
        print("\nSplitting documents into chunks...")
        splits = self.text_splitter.split_documents(langchain_docs)
        print(f"Created {len(splits)} chunks")
        print("Sample chunks:")
        for i, split in enumerate(splits[:3], 1):  # Show first 3 chunks
            print(f"\nChunk {i}:")
            print(f"- Length: {len(split.page_content)} characters")
            print(f"- Preview: {split.page_content[:100]}...")

        # Create or update vector store
        print("\nUpdating vector store...")
        if self.vector_store is None:
            print("- Creating new vector store")
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
        else:
            print("- Adding to existing vector store")
            self.vector_store.add_documents(splits)
            
        print("\n=== Vector Store Update Complete ===")
            
    def similarity_search(self, query: str, k: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of documents with similarity scores
        """
        if self.vector_store is None:
            return []
            
        # Search with scores
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Filter and format results
        filtered_results = []
        for doc, score in results:
            # Convert distance to similarity score (FAISS returns L2 distance)
            similarity = 1.0 / (1.0 + score)
            
            if similarity < score_threshold:
                continue
                
            result = {
                'content': doc.page_content,
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 1),
                'similarity_score': similarity
            }
            filtered_results.append(result)
            
        return filtered_results
        
    def save(self, directory: str = None) -> None:
        """Save vector store to disk.
        
        Args:
            directory: Directory to save the vector store (optional)
        """
        if self.vector_store is not None:
            directory = directory or self.db_path
            directory = Path(directory)
            os.makedirs(directory, exist_ok=True)
            self.vector_store.save_local(str(directory))
            
    @classmethod
    def load(cls, db_path: str, embedding_model: str = "text-embedding-ada-002") -> "VectorStore":
        """Load vector store from disk.
        
        Args:
            db_path: Path to load vector store
            embedding_model: Name of OpenAI embedding model to use
            
        Returns:
            VectorStore instance
        """
        instance = cls(db_path, embedding_model=embedding_model)
        directory = Path(db_path)
        
        # Check for required FAISS index files
        index_file = directory / "index.faiss"
        pkl_file = directory / "index.pkl"
        
        if directory.exists() and index_file.exists() and pkl_file.exists():
            try:
                instance.vector_store = FAISS.load_local(
                    str(directory),
                    instance.embeddings,
                    allow_dangerous_deserialization=True  # Only enable this for trusted local files
                )
            except Exception as e:
                print(f"Error loading vector store: {e}")
                instance.vector_store = None
        else:
            print(f"No valid vector store found at {directory}")
            instance.vector_store = None
            
        return instance
        
    def clear(self) -> None:
        """Clear all vectors from store."""
        self.vector_store = None
        
    @property
    def is_empty(self) -> bool:
        """Check if the vector store is empty."""
        return self.vector_store is None or len(self.vector_store.docstore._dict) == 0
    
    def get_stored_files(self) -> List[Dict[str, Any]]:
        """Get list of files stored in the knowledge base.
        
        Returns:
            List of dictionaries containing file information (name, pages)
        """
        if self.vector_store is None or len(self.vector_store.docstore._dict) == 0:
            return []
            
        files = {}
        for doc_id, doc in self.vector_store.docstore._dict.items():
            source = doc.metadata.get('source', '')
            page = doc.metadata.get('page', 1)
            
            if source in files:
                files[source]['pages'].add(page)
            else:
                files[source] = {
                    'name': os.path.basename(source),
                    'path': source,
                    'pages': {page}
                }
        
        # Convert to list and sort pages
        result = []
        for file_info in files.values():
            file_info['pages'] = sorted(list(file_info['pages']))
            result.append(file_info)
            
        return sorted(result, key=lambda x: x['name'])
