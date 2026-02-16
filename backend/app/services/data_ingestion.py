import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
PDF_FILE_PATH = "D:\\NyayShayak\\Data\\ipc_bns.pdf" 
CHROMA_DB_DIR = "./chroma_db"            
COLLECTION_NAME = "ipc_bns_statutes"  

def load_and_chunk_pdf(file_path: str):
    """Loads the PDF and splits it into manageable chunks for the AI."""
    print(f"Loading document: {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the file at {file_path}. Please check the path.")

    # Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    # Chunking Strategy: 
    # We use a relatively large chunk size (1000) with a healthy overlap (200) 
    # to ensure that if a legal section spans across two paragraphs, the context isn't lost.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Tries to split at paragraphs first
    )

    chunks = text_splitter.split_documents(documents)
    print(f" Split document into {len(chunks)} chunks.")
    return chunks

def ingest_to_vector_db(chunks):
    """Converts text chunks to embeddings and stores them in ChromaDB."""
    print("Initializing embedding model (this may take a moment to download on first run)...")
    
    # We use a fast, lightweight embedding model perfect for CPU processing
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Saving to ChromaDB collection: '{COLLECTION_NAME}'...")
    
    # Create or update the vector store
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME
    )
    
    print("Ingestion complete! The statute database is ready to be queried.")
    return vector_db

if __name__ == "__main__":
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    
    try:
        # 1. Process the PDF
        document_chunks = load_and_chunk_pdf(PDF_FILE_PATH)
        
        # 2. Save to Database
        ingest_to_vector_db(document_chunks)
        
    except Exception as e:
        print(f" An error occurred: {e}")