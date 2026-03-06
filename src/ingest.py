import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
PDF_PATH = "data/report1.pdf"
DB_PATH = "chroma_db"  

def ingest_data():
    """
    Ingests data from a PDF file and saves it to a local database.

    The database is created using the Chroma library, which is
    a vector database that stores embeddings of text documents.

    The ingestion process involves the following steps:

    1. Load the PDF file using the PyPDFLoader
    2. Split the text into chunks using the RecursiveCharacterTextSplitter
    3. Embed the chunks using the HuggingFaceEmbeddings model
    4. Save the embeddings to a local database using Chroma

    The local database is saved to the path specified in the DB_PATH
    variable.

    This function assumes that the PDF file is located in the
    same directory as the script, and that the database should be
    saved in the same directory.

    return: None
    """
    print(f"Loading {PDF_PATH}...")
    
    loader = PyPDFLoader(PDF_PATH)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} pages.")

    # SPLIT TEXT 
    # TEMP: 1000 characters per chunk, with 200 characters overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""] # Try to split by paragraphs first
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(chunks)} chunks.")

    # EMBED & SAVE 
    print("Creating embeddings... (This might take a minute)")
    
    # We use all-MiniLM-L6-v2, TODO: change into BAAI/bge-base-en-v1.5 for better retrieval (FREE, local embedding model)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # This creates the database on your hard drive
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    
    print(f"✅ Success! Database saved to {DB_PATH}")

if __name__ == "__main__":
    ingest_data()