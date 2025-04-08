import os
import streamlit as st
from io import StringIO
from typing import List, Dict, Any, Literal, Optional
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from PyPDF2 import PdfReader
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)

# Configuration
db_type = "support_collection"
store_path = "./tmp_pdf/pdf_store.faiss"
embedding_model = "all-MiniLM-L6-v2"

# Session state initialization
def init_session_state():
    print("Session initialized successfully.")
    if 'databases' not in st.session_state:
        st.session_state.databases = {}
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None

init_session_state()

# Initialize models
def initialize_models():
    try:
        # Local embeddings with HuggingFaceEmbeddings (replacing SentenceTransformerEmbeddings)
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        print("Embeddings initialized successfully.")

        # Cloud-based LLM: Grok
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        st.session_state.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name="qwen-2.5-32b" , #"mistral-saba-24b",
            temperature=0
        )
        print("LLM initialized successfully.")

        # Load existing FAISS index if available
        if os.path.exists(store_path):
            print(f"Loading vector store from {store_path}")
            st.session_state.databases[db_type] = FAISS.load_local(
                store_path,
                st.session_state.embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        return True
    except Exception as e:
        print(f"111 {e}")
        st.error(f"Failed to initialize models: {str(e)}")
        st.session_state.embeddings = None
        st.session_state.llm = None
        return False

# Process uploaded PDFs
def process_document(file) -> List[Document]:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        os.unlink(tmp_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        texts = text_splitter.split_documents(documents)
        return texts
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return []

# Query the database
def query_database(db: FAISS, question: str) -> tuple[str, list]:
    try:
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        relevant_docs = retriever.get_relevant_documents(question)

        if relevant_docs:
            retrieval_qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer concisely based on the provided context. If the context lacks sufficient information, say so."),
                ("human", "Context: {context}\nQuestion: {input}\nAnswer:"),
            ])
            combine_docs_chain = create_stuff_documents_chain(st.session_state.llm, retrieval_qa_prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
            
            response = retrieval_chain.invoke({"input": question})

            context_text = "\n".join([doc.page_content for doc in relevant_docs])
            prompt_text = retrieval_qa_prompt.format(context=context_text, input=question)
            answer_text = response['answer']

   
            # Rough token estimation (e.g., 1 token ≈ 4 characters)
            prompt_tokens = len(prompt_text) // 4
            completion_tokens = len(answer_text) // 4
            token_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }

            return response['answer'], relevant_docs, token_usage
        
        return "No relevant information found in the documents.", []
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "I encountered an error. Please try again.", []

# Main application
def main():
    st.set_page_config(page_title="Document Analyzer", layout="wide", page_icon="📚")
   # st.title("Agent with Database")
    

    with st.sidebar:
        st.header("Document Upload")
        st.info("Upload PDFs to populate the database.")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type="pdf",
            key=f"upload_{db_type}",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with st.spinner('Processing documents...'):
                if not st.session_state.embeddings:
                    st.error("Embeddings not initialized. Cannot process documents.")
                    return
                
                all_texts = []
                for uploaded_file in uploaded_files:
                    texts = process_document(uploaded_file)
                    all_texts.extend(texts)
                
                if all_texts:
                    try:
                        vector_store = FAISS.from_documents(all_texts, st.session_state.embeddings)
                        vector_store.save_local(store_path)
                        st.session_state.databases[db_type] = vector_store
                        st.success("Documents processed and added to the database!")
                    except AttributeError as e:
                        st.error(f"Error creating vector store: {str(e)}. Embeddings may not be initialized.")
            # Initialize models at the start
        if not st.session_state.embeddings or not st.session_state.llm:
            
            if initialize_models():
                st.success("Connected to models and DB successfully!")
            else:
                st.error("Failed to initialize. Check your Grok API key and dependencies.")
                return
            
    url = "https://saffron-edge.getoutline.com/collection/saffron-sales-3r0p0r1s5w/recent"
    st.markdown("Content details [click here](%s)" % url)
    st.header("Ask Questions")
    st.info("Ask a question based on the uploaded documents.")
    question = st.text_input("Enter your question:")
    
    if question:
        with st.spinner('Finding answer...'):
            db = st.session_state.databases.get(db_type)
            if db and st.session_state.llm:
                answer, relevant_docs,token_usage = query_database(db, question)

                st.write("### Answer")
                st.write(answer)
                st.text(f' token used : {token_usage['total_tokens']}')
            else:
                st.error("No database or LLM available. Please upload documents and ensure models are initialized.")

if __name__ == "__main__":
    main()