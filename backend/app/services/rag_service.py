from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil
import os

PERSIST_DIRECTORY = "E:\\case_database"

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

async def store_case_data(full_text: str, case_id: str):
   
    # 1. Clean previous DB for this demo (Optional: In prod, keep them)
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
        except:
            pass

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.create_documents([full_text])

    # 3. Create Vector Store
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )
    # db.persist() # New versions of Chroma persist automatically
    
    return f"Stored {len(chunks)} chunks for Case {case_id}"

def get_relevant_context(query: str):
    """
    Retrieves the 3 most relevant text chunks for a specific query.
    """
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY, 
        embedding_function=embedding_function
    )
    
    # Search for top 3 relevant chunks
    results = db.similarity_search(query, k=3)
    
    # Combine them into one string
    context_text = "\n\n".join([doc.page_content for doc in results])
    return context_text