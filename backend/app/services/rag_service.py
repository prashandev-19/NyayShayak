from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch
import os

PERSIST_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db"))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initializing Embeddings on: {device.upper()}")

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': device}
)

async def store_case_data(full_text: str, case_id: str):
    """
    Chunks an FIR and stores it in the 'case_files' collection 
    with the case_id attached as metadata.
    """
    try:
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.create_documents(
            texts=[full_text], 
            metadatas=[{"case_id": case_id}]
        )

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=PERSIST_DIRECTORY,
            collection_name="case_files" 
        )
        
        return f"Stored {len(chunks)} chunks for Case {case_id}"
    except Exception as e:
        raise Exception(f"Failed to store case data for {case_id}: {str(e)}")


async def get_relevant_context(query: str, collection_name: str = "case_files", filter: dict = None, top_k: int = 3):
    """
    Retrieves the most relevant text chunks. Can query either the case files 
    or the law statutes depending on the collection_name provided.
    """
    try:
        if not os.path.exists(PERSIST_DIRECTORY):
            raise FileNotFoundError(f"Persist directory not found: {PERSIST_DIRECTORY}")
        
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY, 
            embedding_function=embedding_function,
            collection_name=collection_name 
        )
        
        results = db.similarity_search(
            query=query, 
            k=top_k, 
            filter=filter 
        )
        
        context_chunks = [doc.page_content for doc in results]
        
        return context_chunks
    except Exception as e:
        raise Exception(f"Failed to retrieve context for query '{query}': {str(e)}")