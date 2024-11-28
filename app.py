import os
import streamlit as st

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain.docstore.document import Document

class GoogleGenerativeAIEmbeddings(Embeddings):
    """Custom Embeddings class for Google Generative AI"""
    def __init__(self, api_key):
        try:
            genai.configure(api_key=api_key)
            self.model = 'models/embedding-001'
        except Exception as e:
            st.error(f"Error configuring embeddings: {e}")
            self.model = None

    def embed_documents(self, texts):
        """Embed a list of documents"""
        if not self.model:
            raise ValueError("Embedding model not initialized")
        
        try:
            embeddings = []
            for text in texts:
                # Truncate text if too long
                result = genai.embed_content(
                    model=self.model,
                    content=text,  # Limit to first 1000 characters
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        except Exception as e:
            st.error(f"Embedding error: {e}")
            return []

    def embed_query(self, text):
        """Embed a query"""
        if not self.model:
            raise ValueError("Embedding model not initialized")
        
        try:
            result = genai.embed_content(
                model=self.model,
                content=text[:1000],  # Limit to first 1000 characters
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            st.error(f"Query embedding error: {e}")
            return []

class RepositoryRAGChatbot:
    def __init__(self):
        # Initialize session state
        if 'readmes' not in st.session_state:
            st.session_state.readmes = []
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # API Key setup
        self.api_key = "AIzaSyAxykNQtukQdVC1MPE1TakumduPPk93h6A"
        
        # Initialize models
        try:
            genai.configure(api_key=self.api_key)
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=self.api_key, 
                model="gemini-pro"
            )
            self.embedding = GoogleGenerativeAIEmbeddings(self.api_key)
        except Exception as e:
            st.error(f"Model initialization error: {e}")
            self.llm = None
            self.embedding = None

    def find_readme_files(self, base_path):
        """
        Recursively find README files in all subdirectories and files with .md extension.
        Returns a list of dictionaries with file details.
        
        :param base_path: The root directory to start the search.
        :return: A list of dictionaries with details of README and .md files.
        """
        readme_files = []

        # Accepted README filenames and extensions
        valid_readme_names = ["readme.md", "readme.txt", "readme.adoc", "readme.rst"]

        # Walk through the directory tree
        for root, dirs, files in os.walk(base_path):
            for file in files:
                # Check if the filename matches a known README or ends with .md
                if file.lower() in valid_readme_names or file.lower().endswith(".md"):
                    full_path = os.path.join(root, file)

                    try:
                        # Read the content of the file
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # Prepare the file details
                        file_info = {
                            'path': full_path,
                            'name': file,
                            'relative_path': os.path.relpath(full_path, base_path),
                            'content': content
                        }

                        readme_files.append(file_info)

                    except Exception as e:
                        print(f"Warning: Could not read {full_path}: {e}")

        return readme_files

    def create_vector_store(self, selected_readmes):
        """Create vector store from selected README files"""
        try:
            # Convert to Langchain documents
            documents = [
                Document(
                    page_content=readme['content'], 
                    metadata={
                        'source': readme['relative_path']
                    }
                ) for readme in selected_readmes
            ]
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(documents)
            
            # Create FAISS vector store
            vector_store = FAISS.from_documents(
                split_docs, 
                self.embedding
            )
            
            return vector_store
        
        except Exception as e:
            st.error(f"Vector store creation error: {e}")
            return None

    def retrieve_context(self, vector_store, query):
        """Retrieve relevant context from vector store"""
        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            context_docs = retriever.get_relevant_documents(query)
            return " ".join([doc.page_content for doc in context_docs])
        except Exception as e:
            st.error(f"Context retrieval error: {e}")
            return ""

    def generate_response(self, query, context):
        """Generate response using Gemini"""
        try:
            prompt = f"""Context from README files:
            {context}

            User Query: {query}

            Use the context to help answer the query. If the context doesn't contain 
            sufficient information, answer based on your general knowledge."""

            response = self.llm.invoke(prompt)
            return response.content
        
        except Exception as e:
            st.error(f"Response generation error: {e}")
            return "Sorry, I couldn't generate a response."

    def run(self):
        """Main Streamlit application"""
        st.title("Multi-README Repository RAG Chatbot")
        
        # Sidebar for repository path and file selection
        with st.sidebar:
            st.header("Repository Configuration")
            
            # Repository path input
            repo_path = st.text_input("Enter Repository Path")
            
            # Find README files button
            if st.button("Find README Files"):
                try:
                    # Find README files
                    st.session_state.readmes = self.find_readme_files(repo_path)
                    
                    if st.session_state.readmes:
                        st.success(f"Found {len(st.session_state.readmes)} README files")
                    else:
                        st.warning("No README files found")
                
                except Exception as e:
                    st.error(f"Error finding README files: {e}")
            
            # Multi-select README files if found
            if hasattr(st.session_state, 'readmes') and st.session_state.readmes:
                selected_files = st.multiselect(
                    "Select README files to include in RAG",
                    options=[readme['relative_path'] for readme in st.session_state.readmes],
                    default=None
                )
                
                # Create vector store from selected files
                if st.button("Load Selected Files"):
                    # Filter selected files
                    selected_readmes = [
                        readme for readme in st.session_state.readmes 
                        if readme['relative_path'] in selected_files
                    ]
                    
                    # Create vector store
                    vector_store = self.create_vector_store(selected_readmes)
                    
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.success(f"Loaded {len(selected_readmes)} README files into vector store")
        
        # Main chat interface
        if hasattr(st.session_state, 'vector_store'):
            # Chat input
            user_query = st.chat_input("Ask a question about the repository")
            
            if user_query:
                # Add user message to chat history
                st.session_state.messages.append({
                    "role": "user", 
                    "content": user_query
                })
                
                # Retrieve context
                context = self.retrieve_context(
                    st.session_state.vector_store, 
                    user_query
                )
                
                # Generate response
                response = self.generate_response(user_query, context)
                
                # Add response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

def main():
    chatbot = RepositoryRAGChatbot()
    chatbot.run()

if __name__ == "__main__":
    main()