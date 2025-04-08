# Document Analyzer Application

## Overview
The application allows users to upload PDF documents, processes them into a searchable vector database, and enables users to ask questions based on the content of the uploaded documents. It leverages natural language processing (NLP) and retrieval-augmented generation (RAG) techniques to provide concise answers derived from the document contents.

The application uses a combination of local embeddings (via HuggingFace) and a cloud-based large language model (LLM) from Groq to process and query the documents. The vector database is implemented using FAISS for efficient similarity search.

## What the Application Does

### Model Initialization
- Local embedding model (`all-MiniLM-L6-v2`) using `HuggingFaceEmbeddings`.
- Connects to a cloud-based LLM (`qwen-2.5-32b`) via the Groq API.
- Loads an existing FAISS vector store if available or creates a new one from uploaded documents.

### Document Processing
- Accepts PDF uploads from users via Streamlit’s file uploader.
- Extracts text from PDFs.
- Splits the text into smaller chunks.
- Converts text chunks into vector embeddings and stores them in a FAISS vector store.

### Query Processing
- Takes a user-provided question as input.
- Retrieves relevant document chunks from the FAISS vector store using similarity search.
- Uses a retrieval chain with a custom prompt and the Groq LLM to generate a concise answer based on the retrieved context.
- Estimates token usage for the prompt and response.

### User Interface
- Provides a sidebar for uploading PDFs.
- Displays a main section for entering questions and viewing answers.
- Includes status messages (e.g., "Processing documents...", "Finding answer...") and error handling.

## Tools and Libraries Used

### Libraries
Below is the list of libraries used for AI processes in the code, with their names, purposes, and an indication of whether they are paid or free. Note that "paid" refers to whether the library itself or its associated services (e.g., APIs) require payment.

- **`langchain_huggingface`**
  - **Purpose**: Generates vector embeddings for text using Hugging Face models.
  - **Paid/Free**: Free (open-source library; the `all-MiniLM-L6-v2` model from Hugging Face is also freely available for local use).

- **`langchain_groq`**
  - **Purpose**: Integrates with Groq’s cloud-based LLM for natural language generation.
  - **Paid/Free**: Free (open-source library), but requires a paid Groq API key for accessing the Groq cloud service (e.g., the `qwen-2.5-32b` model). Groq offers a free tier with limits, but full usage typically involves a paid subscription.

- **`langchain_community`**
  - **Purpose**: Provides tools for document loading (`PyPDFLoader`) and vector storage (`FAISS`).
  - **Paid/Free**: Free (open-source library; both `PyPDFLoader` and `FAISS` are freely available).

- **`langchain`**
  - **Purpose**: Offers components for text splitting, prompt templating, and retrieval-augmented generation pipelines.
  - **Paid/Free**: Free (open-source library).

- **`langchain_core`**
  - **Purpose**: Supplies foundational classes like `Document` for structuring text data.
  - **Paid/Free**: Free (open-source library).

### Notes on Paid Components
The only paid element in this setup is the Groq API accessed via `langchain_groq`. While the library itself is free, using Groq’s cloud-based LLM (e.g., `qwen-2.5-32b`) requires an API key, and Groq’s pricing depends on usage beyond their free tier. All other libraries are open-source and free to use without additional costs.

## How It Works

### Startup
- The app initializes session state and attempts to connect to the embedding model and LLM.
- If a FAISS index exists, it loads it into memory.

### Document Upload
- Users upload PDFs via the sidebar.
- The PDFs are processed into text chunks, embedded, and stored in a FAISS vector store.

### Question Answering
- Users enter a question in the main interface.
- The app retrieves relevant document chunks, sends them to the LLM with the question, and displays the answer.

### Error Handling
- Errors during model initialization, document processing, or querying are displayed to the user via Streamlit’s `st.error`.

## Example Usage
- Upload a PDF containing product documentation.
- Ask, “What are the key features of Product X?”
- The app retrieves relevant sections from the PDF and provides a concise answer.

## Limitations
- Requires a valid `GROQ_API_KEY` to function.
- Only processes PDFs (no support for other file types).
- Token usage estimation is rough (based on character count).
- The FAISS index is stored locally and may grow large with many documents.