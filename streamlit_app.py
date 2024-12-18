import streamlit as st
from generator import QAGenerator
import os
from datetime import datetime
import json
from generator import DEFAULT_SYSTEM_PROMPT
from typing import List, Dict
import time  # Add import for time module

# Initialize session state
if 'qa_logs' not in st.session_state:
    print("Initializing qa_logs at top level")
    st.session_state.qa_logs = []
if 'current_role' not in st.session_state:
    st.session_state.current_role = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'question' not in st.session_state:
    st.session_state.question = ''


def initialize_qa():
    """Initialize QA system if not already done."""
    if 'qa' not in st.session_state:
        st.session_state.qa = QAGenerator(db_path="lecture_db")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None

def log_interaction(question: str, answer: str, sources: list, has_sources: bool):
    """Log Q&A interaction with timestamp."""
    print("\n=== log_interaction START ===")
    print("Input parameters:")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
    print(f"Has sources: {has_sources}")
    
    print("\nSession state before logging:")
    print(f"Session state keys: {st.session_state.keys()}")
    print(f"qa_logs exists: {'qa_logs' in st.session_state}")
    if 'qa_logs' in st.session_state:
        print(f"Current qa_logs length: {len(st.session_state.qa_logs)}")
        print(f"Current qa_logs: {st.session_state.qa_logs}")
    
    # Ensure qa_logs exists
    if 'qa_logs' not in st.session_state:
        print("\nInitializing qa_logs in log_interaction")
        st.session_state.qa_logs = []
    
    # Create log entry
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'question': question,
        'answer': answer,
        'sources': sources,
        'has_sources': has_sources
    }
    print(f"\nCreated log entry: {log_entry}")
    
    # Add to logs
    print("\nUpdating session state:")
    current_logs = list(st.session_state.qa_logs)  # Create a copy
    print(f"Current logs before append: {current_logs}")
    current_logs.append(log_entry)
    print(f"Current logs after append: {current_logs}")
    st.session_state.qa_logs = current_logs  # Update session state
    
    print("\nSession state after update:")
    print(f"qa_logs exists: {'qa_logs' in st.session_state}")
    print(f"qa_logs length: {len(st.session_state.qa_logs)}")
    print(f"qa_logs content: {st.session_state.qa_logs}")
    print("=== log_interaction END ===\n")
    
    # Force Streamlit to recognize the state change
    st.session_state.qa_logs = st.session_state.qa_logs.copy()

def handle_question(question: str):
    """Handle question submission and generate response."""
    if not question:
        return
    
    print("\n=== handle_question START ===")
    print(f"Processing question: {question}")
    print(f"Current session state keys: {st.session_state.keys()}")
    print(f"qa_logs exists: {'qa_logs' in st.session_state}")
    print(f"Current qa_logs: {st.session_state.qa_logs if 'qa_logs' in st.session_state else 'Not found'}")
    
    st.session_state.processing = True
    st.session_state.current_question = question

    try:
        # Get answer and sources
        print("\nCalling generate_answer...")
        result = st.session_state.qa.generate_answer(question)
        print(f"\nResult received from generate_answer: {result}")
        
        # Extract answer and sources, with defaults if missing
        answer = result.get('answer', "No answer generated")
        sources = result.get('sources', [])
        has_sources = bool(sources)
        
        print(f"\nProcessed result:")
        print(f"Answer: {answer}")
        print(f"Has sources: {has_sources}")
        print(f"Sources: {sources}")
        
        # Always log the interaction, even if no sources
        print("\nCalling log_interaction...")
        log_interaction(question, answer, sources, has_sources)
        print("\nAfter log_interaction:")
        print(f"qa_logs exists: {'qa_logs' in st.session_state}")
        print(f"Current qa_logs: {st.session_state.qa_logs if 'qa_logs' in st.session_state else 'Not found'}")
        
        # Display answer
        st.markdown("**Answer:**")
        st.write(answer)
        
        # Display sources if available
        if sources:
            st.markdown("**Sources:**")
            for source in sources:
                st.markdown(f"- {os.path.basename(source['source'])} (Page {source['page']}, Score: {source['score']:.2f})")
        else:
            st.warning("No relevant sources found in the course materials.")
        
        # Generate follow-up questions only if we have sources
        if has_sources and st.session_state.qa:
            follow_up = st.session_state.qa.generate_followup_questions(question, answer)
            if follow_up:
                st.markdown("**Follow-up Questions:**")
                for q in follow_up:
                    if st.button(q, key=f"followup_{hash(q)}"):
                        handle_question(q)
    
    except Exception as e:
        print(f"\nError in handle_question: {str(e)}")
        error_msg = f"Error generating answer: {str(e)}"
        st.error(error_msg)
        
        # Log error cases too
        print("\nLogging error case...")
        log_interaction(question, error_msg, [], False)
        
    finally:
        st.session_state.processing = False
        print("\n=== handle_question END ===")

def student_view():
    """Display student view for asking questions."""
    # Add custom CSS for follow-up question buttons and container
    st.markdown("""
        <style>
        .stButton > button {
            width: 100%;
            height: auto;
            white-space: normal;
            word-wrap: break-word;
            text-align: left;
            padding: 10px;
            background-color: #2b313e;
            color: #ffffff;
            border: 1px solid #404756;
            border-radius: 4px;
            margin: 5px 0;
        }
        .stButton > button:hover {
            background-color: #404756;
            border-color: #4a5568;
            color: #ffffff;
        }
        .stButton > button:active {
            background-color: #404756;
            border-color: #4a5568;
            color: #ffffff !important;
        }
        #chat-container {
            margin-bottom: 100px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📚 Course Q&A Assistant")
    
    # Initialize QA system if not already done
    initialize_qa()
    
    # Initialize session states
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Check if vector store is empty
    if st.session_state.qa.vector_store.is_empty:
        st.warning("No course materials have been uploaded yet. Please contact your instructor.")
        return
    
    # Create a container for chat messages
    chat_container = st.container()
    
    # Question input at the bottom
    with st.container():
        st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
        question = st.chat_input("Ask a question about the course materials")
    if question:
            st.session_state.current_question = question
            st.session_state.processing = True
            st.rerun()
    
    # Display chat history in the container
    with chat_container:
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                # Display message content
                st.write(message["content"])
                
                # Display sources for assistant messages
                if message["role"] == "assistant":
                    if "sources" in message and message["sources"]:
                        st.divider()
                        st.write("📚 **Sources:**")
                        for source in message["sources"]:
                            score = source.get('score', 0)
                            page = source.get('page', '')
                            filename = os.path.basename(source.get('source', ''))
                            page_str = f" (Page {page})" if page else ""
                            score_str = f" (Score: {score:.2f})" if score > 0 else ""
                            st.write(f"- {filename}{page_str}{score_str}")
                            if 'content' in source:
                                with st.expander("View Source Content"):
                                    st.write(source['content'])
                    
                    # Show follow-up questions
                    if "followup_questions" in message and message["followup_questions"]:
                        st.divider()
                        st.write("🤔 **Follow-up Questions:**")
                        for i, q in enumerate(message["followup_questions"]):
                            # Create unique key using message index and question index
                            button_key = f"followup_msg{idx}_q{i}"
                            if st.button(q, key=button_key):
                                st.session_state.current_question = q
                                st.session_state.processing = True
                                st.rerun()
    
    # Process current question if needed
    if st.session_state.processing and st.session_state.current_question:
        try:
            print("\n=== Processing Question in student_view ===")
            print(f"Question: {st.session_state.current_question}")
            
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": st.session_state.current_question
            })
            
            # Generate response
            with st.spinner("Thinking..."):
                response = st.session_state.qa.generate_answer(st.session_state.current_question)
                print(f"\nResponse from generate_answer: {response}")
                
                # Log the interaction
                has_sources = bool(response.get('sources', []))
                print("\nLogging interaction...")
                log_interaction(
                    question=st.session_state.current_question,
                    answer=response['answer'],
                    sources=response.get('sources', []),
                    has_sources=has_sources
                )
                
                followup_questions = st.session_state.qa.generate_followup_questions(
                    question=st.session_state.current_question,
                    answer=response['answer'],
                    k=3
                )
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response['answer'],
                "sources": response['sources'],
                "followup_questions": followup_questions
            })
            
        except Exception as e:
            print(f"\nError in student_view: {str(e)}")
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            
            # Add error message
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": []
            })
            
            # Log error case
            print("\nLogging error case...")
            log_interaction(
                question=st.session_state.current_question,
                answer=error_msg,
                sources=[],
                has_sources=False
            )
        
        # Reset processing state
        st.session_state.current_question = None
        st.session_state.processing = False
        st.rerun()
    
    # Clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.current_question = None
        st.session_state.processing = False
        st.session_state.qa.clear_conversation_history()
        st.rerun()

def instructor_view():
    """Display instructor view for uploading content."""
    st.title("📚 Course Q&A Assistant - Instructor Dashboard")
    
    # Initialize QA system and session state
    initialize_qa()
    
    # Content Management
    st.header("📑 Knowledge Base Management")
    
    # Display knowledge base status
    if hasattr(st.session_state, 'qa'):
        with st.expander("Knowledge Base Status", expanded=True):
            if st.session_state.qa.vector_store.is_empty:
                st.warning("Knowledge base is empty. Please upload course materials.")
            else:
                st.success("Knowledge base is populated with the following materials:")
                
                # Show stored files
                files = st.session_state.qa.vector_store.get_stored_files()
                for file in files:
                    with st.container():
                        col1, col2 = st.columns([3, 7])
                        with col1:
                            st.write(f"📄 **{file['name']}**")
                        with col2:
                            pages = file['pages']
                            if len(pages) == 1:
                                st.write(f"Page: {pages[0]}")
                            else:
                                st.write(f"Pages: {min(pages)}-{max(pages)}")
    
    st.markdown("---")
    
    # File Upload Section
    st.subheader("Add Course Materials")
    uploaded_files = st.file_uploader(
        "Upload PDF or Jupyter notebook files", 
        type=['pdf', 'ipynb'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process Files"):
        with st.spinner("Processing files..."):
            # Save uploaded files temporarily
            file_paths = []
            for file in uploaded_files:
                temp_path = f"temp_{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.getvalue())
                file_paths.append(temp_path)
                
            try:
                # Process files
                st.session_state.qa.add_documents(file_paths)
                st.success("Files processed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
            finally:
                # Clean up temporary files
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        os.remove(file_path)
    
    st.markdown("---")
    
    # Knowledge Update Section
    st.header("🔄 Knowledge Updates")
    
    # Display Q&A logs
    st.subheader("Recent Q&A Interactions")
    print("\n=== instructor_view: Displaying Q&A Logs ===")
    print(f"Session state keys: {st.session_state.keys()}")
    print(f"qa_logs exists: {'qa_logs' in st.session_state}")
    print(f"Number of logs: {len(st.session_state.qa_logs) if 'qa_logs' in st.session_state else 0}")
    print(f"Logs in session state: {st.session_state.qa_logs if 'qa_logs' in st.session_state else 'Not found'}")
    
    if st.session_state.qa_logs:
        print("\nDisplaying logs:")
        for log in reversed(st.session_state.qa_logs):
            with st.expander(f"Q: {log['question']}", expanded=False):
                st.write("**Question:**")
                st.write(log['question'])
                st.write("**Answer:**")
                st.write(log['answer'])
                if log['sources']:
                    st.write("**Sources:**")
                    for source in log['sources']:
                        st.write(f"- {os.path.basename(source['source'])} (Page {source['page']}, Score: {source['score']:.2f})")
                else:
                    st.warning("No relevant sources found")
                st.write(f"**Time:** {log['timestamp']}")
                
                # Show suggestion for questions with no sources
                if not log['sources']:
                    st.divider()
                    st.write("🤔 **Knowledge Base Update Suggestion:**")
                    
                    # Generate suggestion if not already in session state
                    suggestion_key = f"suggestion_{hash(log['question'])}"
                    update_key = f"updated_{hash(log['question'])}"
                    
                    if suggestion_key not in st.session_state:
                        with st.spinner("Generating suggestion..."):
                            suggestion = st.session_state.qa.generate_knowledge_suggestion(log['question'])
                            st.session_state[suggestion_key] = suggestion
                    
                    # Show suggestion text area (disabled if already updated)
                    is_updated = st.session_state.get(update_key, False)
                    suggestion = st.text_area(
                        "Suggested content to add:",
                        value=st.session_state[suggestion_key],
                        height=200,
                        key=f"edit_{suggestion_key}",
                        disabled=is_updated
                    )
                    
                    # Add update button (hidden if already updated)
                    if not is_updated:
                        if st.button("Update Knowledge Base", key=f"update_{hash(log['question'])}"):
                            try:
                                with st.spinner("Updating knowledge base..."):
                                    # Add text directly to knowledge base
                                    source = f"manual_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                    success = st.session_state.qa.add_text_to_knowledge(
                                        suggestion, 
                                        source=source
                                    )
                                    
                                    if success:
                                        st.session_state[update_key] = True
                                        st.success("✅ Knowledge base updated successfully!")
                                    else:
                                        st.error("Failed to update knowledge base")
                                        
                            except Exception as e:
                                st.error(f"Error updating knowledge base: {str(e)}")
                    else:
                        st.success("✅ This suggestion has been added to the knowledge base")
    else:
        st.info("No Q&A logs available yet")
    
    st.markdown("---")
    
    # System Prompt Management
    st.header("🤖 QA Bot Configuration")
    with st.expander("Configure QA Bot Behavior", expanded=True):
        # Initialize system prompt in session state if not exists
        if 'system_prompt' not in st.session_state:
            st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
        
        st.markdown("""
        Customize how the QA bot should behave when answering student questions. 
        You can specify rules such as:
        - Focus on theoretical concepts
        - Stay strictly within course material
        - Include specific types of examples
        - Format requirements for answers
        """)
        
        new_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            height=200,
            help="Define how the QA bot should behave when answering questions"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Update Prompt"):
                st.session_state.system_prompt = new_prompt
                st.session_state.qa.update_system_prompt(new_prompt)
                st.success("QA bot configuration updated!")
        with col2:
            if st.button("Reset to Default"):
                st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
                st.session_state.qa.update_system_prompt(DEFAULT_SYSTEM_PROMPT)
                st.success("Reset to default configuration!")

def main():
    """Main application."""
    # Initialize QA system
    initialize_qa()
    
    # Role selection in sidebar
    st.sidebar.title("🎓 LectureGPT")
    role = st.sidebar.radio("Select Role:", ["Student", "Instructor"])
    
    if role == "Student":
        student_view()
    else:
        instructor_view()
        
if __name__ == "__main__":
    main()
