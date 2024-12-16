from typing import List, Dict, Any
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from vector_store import VectorStore
from retriever import Retriever
from parsing import file2content, get_enhance_content
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_SYSTEM_PROMPT = """You are a helpful teaching assistant. Use the following context and conversation history to answer the student's question.
If the answer cannot be found in the context, say so clearly.
Focus on providing accurate information from the course materials.
Do not make up or infer information that is not supported by the context."""

class QAGenerator:
    def __init__(self, db_path: str = "lecture_db", system_prompt: str = None):
        """Initialize QA generator with vector store and conversation memory.
        
        Args:
            db_path: Path to vector store database
            system_prompt: Custom system prompt for the QA bot
        """
        self.db_path = db_path  # Store db_path as instance variable
        try:
            self.vector_store = VectorStore.load(db_path)
        except:
            print(f"Could not load vector store from {db_path}")
            self.vector_store = VectorStore(db_path)
            
        self.retriever = Retriever(self.vector_store)
        
        # Initialize conversation components
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.llm = ChatOpenAI(temperature=0)
        
        # Set system prompt
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.update_prompt_template()
        
        # Initialize LLM chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=True
        )
    
    def update_prompt_template(self):
        """Update the prompt template with current system prompt."""
        template = f"""{self.system_prompt}

        Context:
        {{context}}

        Conversation History:
        {{chat_history}}

        Student Question: {{question}}

        Assistant Answer:"""
        
        self.prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=template
        )
        
        # Update chain with new prompt
        if hasattr(self, 'chain'):
            self.chain.prompt = self.prompt
    
    def update_system_prompt(self, new_prompt: str):
        """Update the system prompt and refresh the chain.
        
        Args:
            new_prompt: New system prompt to use
        """
        self.system_prompt = new_prompt
        self.update_prompt_template()
        
    def add_documents(self, file_paths: List[str]) -> None:
        """Add new documents to the system.
        
        Args:
            file_paths: List of file paths to process
        """
        print(f"Input file_paths: {file_paths}")
        
        # Get content from files
        contents = file2content(file_paths)
        if not contents:
            print("No contents extracted from files")
            return
            
        print(f"Contents length: {len(contents)}")
        print(f"First content item: {contents[0] if contents else 'No contents'}")
        
        # Enhance content
        enhanced = get_enhance_content(contents)
        if not enhanced:
            print("No enhanced content generated")
            return
            
        print(f"Enhanced length: {len(enhanced)}")
        print(f"Enhanced content samples: {enhanced[:2]}")
        
        # Prepare for vector store
        documents = []
        
        # Process in batches to avoid memory issues
        batch_size = 10
        for i in range(0, len(enhanced), batch_size):
            batch = enhanced[i:i + batch_size]
            batch_contents = contents[i:i + batch_size]
            
            for j, content in enumerate(batch):
                if not content or content == "NONE":
                    print(f"Skipping content {i+j} as it's empty or NONE")
                    continue
                    
                try:
                    # Store metadata based on file type
                    source_content = batch_contents[j]
                    raw_text = source_content.get('raw_text', '')
                    enhanced_text = content
                    
                    document = {
                        "content": enhanced_text,  # Store enhanced content
                        "raw_text": raw_text,     # Keep original text
                        "source": source_content.get("file_path", file_paths[min(i+j, len(file_paths)-1)]),
                        "page": source_content.get("page_number", source_content.get("cell_number", 1))
                    }
                    
                    # Add additional metadata if available
                    if source_content.get("doc_metadata"):
                        document["doc_metadata"] = source_content["doc_metadata"]
                    if source_content.get("cell_type"):
                        document["cell_type"] = source_content["cell_type"]
                    
                    documents.append(document)
                except Exception as e:
                    print(f"Error processing content {i+j}: {str(e)}")
                    continue
            
        print(f"Final documents length: {len(documents)}")
            
        # Add to vector store
        if documents:
            try:
                self.vector_store.add_documents(documents)
                # Save after successful addition
                self.vector_store.save(str(self.db_path))
                print(f"Successfully added {len(documents)} documents to vector store")
            except Exception as e:
                print(f"Error adding to vector store: {str(e)}")
        else:
            print("No valid documents to add to vector store")
        
    def add_text_to_knowledge(self, text: str, source: str = "manual_update") -> bool:
        """Add text directly to knowledge base without file creation.
        
        Args:
            text: Text content to add
            source: Source identifier for the content
            
        Returns:
            bool: True if update was successful
        """
        print(f"\n=== Adding Text to Knowledge Base ===")
        print(f"Text length: {len(text)}")
        
        # Create document for vector store
        document = {
            "content": text,
            "raw_text": text,
            "source": source,
            "page": 1
        }
        
        # Add to vector store
        try:
            print("Adding to vector store...")
            self.vector_store.add_documents([document])
            print("Saving vector store...")
            self.vector_store.save()  # Use default path from VectorStore
            print("Successfully added text to vector store")
            return True
        except Exception as e:
            print(f"Error adding to vector store: {str(e)}")
            return False

    def generate_answer(self, question: str) -> Dict[str, Any]:
        """Generate answer for a question using retrieved context and conversation history.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer and sources
        """
        print("\n=== Generating Answer ===")
        print(f"Question: {question}")
        
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(question)
            print(f"Retrieved {len(retrieved_docs)} documents")
            
            # Format context from retrieved documents
            context = self.retriever.get_formatted_context(retrieved_docs)
            
            # Handle case when no relevant documents found
            if not context:
                print("No relevant context found")
                no_context_response = {
                    'answer': "I couldn't find any relevant information in the course materials to answer your question.",
                    'sources': []
                }
                
                # Save to conversation history even when no context
                self.memory.save_context(
                    {"input": question},
                    {"output": no_context_response['answer']}
                )
                
                return no_context_response
            
            # Get conversation history
            memory_vars = self.memory.load_memory_variables({})
            chat_history = memory_vars.get("chat_history", "")
            
            # Generate answer using LLM chain
            result = self.chain.run({
                "context": context,
                "question": question,
                "chat_history": chat_history
            })
            
            # Save the interaction to memory
            self.memory.save_context(
                {"input": question},
                {"output": result}
            )
            
            # Extract source information
            sources = []
            for doc in retrieved_docs:
                source_info = {
                    'source': doc['source'],
                    'page': doc['page'],
                    'score': doc['similarity_score'],
                    'content': doc.get('content', '')
                }
                sources.append(source_info)
            
            print(f"Generated answer with {len(sources)} sources")
            
            return {
                'answer': result,
                'sources': sources
            }
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            error_response = {
                'answer': "I encountered an error while generating the answer. Please try again.",
                'sources': []
            }
            
            # Save error to conversation history
            self.memory.save_context(
                {"input": question},
                {"output": error_response['answer']}
            )
            
            return error_response
    
    def generate_followup_questions(self, question: str, answer: str, k: int = 3) -> List[str]:
        """Generate follow-up questions based on the previous Q&A.
        
        Args:
            question: The original question
            answer: The answer provided
            k: Number of follow-up questions to generate
            
        Returns:
            List of follow-up questions
        """
        template = f"""Based on this previous Q&A, generate {k} relevant follow-up questions that would help deepen understanding of the topic.
        Make questions specific and directly related to the content discussed.
        
        Previous Question: {question}
        Previous Answer: {answer}
        
        Generate exactly {k} follow-up questions, each on a new line starting with a number and a period (e.g., "1. ")."""
        
        messages = [
            {"role": "system", "content": "You are a helpful teaching assistant generating follow-up questions to deepen student understanding."},
            {"role": "user", "content": template}
        ]
        
        response = self.llm(messages)
        response_text = response.content if hasattr(response, 'content') else response
        
        # Extract questions from the response
        questions = []
        for line in response_text.split('\n'):
            line = line.strip()
            if line and line[0].isdigit() and '. ' in line:
                questions.append(line.split('. ', 1)[1])
                
        return questions[:k]  # Ensure we return exactly k questions

    def generate_knowledge_suggestion(self, question: str) -> str:
        """Generate a suggestion for knowledge base content based on the question.
        
        Args:
            question: The question that couldn't be answered
            
        Returns:
            Suggested content to add to knowledge base
        """
        prompt = f"""Given this question that couldn't be answered from the course materials:
        Question: {question}
        
        Generate a concise, educational response that would be appropriate to add to the course materials.
        The response should:
        1. Be factual and educational
        2. Be written in a clear, academic style
        3. Include key concepts and definitions
        4. Be about 2-3 paragraphs long
        
        Response:"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful teaching assistant creating educational content."},
                {"role": "user", "content": prompt}
            ]
            response = self.llm(messages)
            return response.content
            
        except Exception as e:
            print(f"Error generating knowledge suggestion: {e}")
            return "Error generating suggestion. Please try again."

    def needs_knowledge_update(self, sources: List[Dict]) -> bool:
        """Check if the answer needs knowledge base update based on sources.
        
        Args:
            sources: List of sources used in the answer
            
        Returns:
            True if knowledge update is needed
        """
        # Consider update needed if no sources or low relevance scores
        if not sources:
            return True
            
        # Check if any source has a relevance score above threshold
        threshold = 0.7  # Adjust this threshold as needed
        return not any(source.get('score', 0) > threshold for source in sources)

    def clear_conversation_history(self):
        """Clear the conversation memory."""
        self.memory.clear()
