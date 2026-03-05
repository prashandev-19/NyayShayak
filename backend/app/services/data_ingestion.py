import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

PDF_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "Data", "ipc_bns.pdf")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nyayshayak")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

def load_and_chunk_pdf(file_path: str):
    
    print(f"Loading document: {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the file at {file_path}. Please check the path.")

   
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

   
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""] 
    )

    chunks = text_splitter.split_documents(documents)
    
   
    for chunk in chunks:
        chunk.metadata["collection"] = "ipc_bns_statutes"
    
    print(f"✓ Split document into {len(chunks)} chunks.")
    return chunks

def ingest_to_vector_db(chunks):
    
    print("Initializing embedding model (this may take a moment to download on first run)...")
    
   
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
  
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENVIRONMENT
            )
        )
        print(f"Index '{PINECONE_INDEX_NAME}' created successfully")

    print(f"Saving to Pinecone index: '{PINECONE_INDEX_NAME}'...")
    
  
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    
    print("Ingestion complete! The statute database is ready to be queried.")
    return vectorstore

if __name__ == "__main__":
    try:
        
        document_chunks = load_and_chunk_pdf(PDF_FILE_PATH)
        
        
        ingest_to_vector_db(document_chunks)
        
    except Exception as e:
        print(f" An error occurred: {e}")