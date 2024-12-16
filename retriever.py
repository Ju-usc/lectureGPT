from typing import List, Dict, Any
import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from vector_store import VectorStore

class Retriever:
    def __init__(self, vector_store: VectorStore):
        """Initialize retriever with vector store.
        
        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store
        
        # Initialize LangChain components for query processing
        self.llm = OpenAI(temperature=0.3)
        
        # Prompt for query processing
        self.query_prompt = PromptTemplate(
            input_variables=["query"],
            template="""You are a query reformulation expert. Your task is to convert a user's question into a search query that will help find relevant course content.
            If the query is not related to course content or technical topics, return 'NOT_COURSE_RELATED'.
            Otherwise, generate a concise, technical search query focused on the key concepts.
            Remove any conversational elements and focus on technical terms.

            Original query: {query}
            
            Technical search query:"""
        )
        
        self.query_chain = LLMChain(llm=self.llm, prompt=self.query_prompt)
        
    def _process_query(self, query: str) -> str:
        """Process the query to make it more effective for search.
        
        Args:
            query: Original user query
            
        Returns:
            Processed query or empty string if not course related
        """
        # Process query using LangChain
        result = self.query_chain.run(query=query)
        
        # Check if query is course related
        if result.strip() == 'NOT_COURSE_RELATED':
            print(f"Query '{query}' is not related to course content")
            return ''
            
        processed_query = result.strip()
        print(f"Processed query: {processed_query}")
        return processed_query
        
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with similarity scores
        """
        # Process query
        processed_query = self._process_query(query)
        if not processed_query:
            return []
            
        # Search vector store
        results = self.vector_store.similarity_search(
            processed_query,
            k=k,
            score_threshold=0.7
        )
        
        # Print results for debugging
        print("\nRetrieved documents:")
        for i, doc in enumerate(results):
            print(f"\n{i+1}. Score: {doc['similarity_score']:.3f}")
            print(f"Source: {doc['source']}, Page: {doc['page']}")
            print(f"Content: {doc['content'][:200]}...")
            
        return results

    def get_formatted_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a context string.
        
        Args:
            results: List of retrieved documents with metadata
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
            
        context_parts = []
        for doc in results:
            # Format source information
            source_info = f"[Source: {doc['source']}"
            if 'page' in doc:
                source_info += f", Page {doc['page']}"
            if 'similarity_score' in doc:
                source_info += f", Similarity: {doc['similarity_score']:.2f}"
            source_info += "]"
            
            # Get content
            content = doc.get('content', '').strip()
            if not content:
                continue
                
            context_parts.append(f"{source_info}\n{content}")
            
        return "\n\n".join(context_parts)
