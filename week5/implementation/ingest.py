import glob
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Define global constants
LLM_MODEL = "gpt-5-nano"
EMBEDDING_MODEL = "text-embedding-3-small"
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")
DB_NAME = "vector_db"

load_dotenv(override=True)

def fetch_documents() -> list[Document]:
    """
    Read all documents from knowledge-base folder and return a list of documents
    and it's document type (as per folder name) as metadata.

    Args:
        None
    Returns:
        list[Document]: List of documents
    """
    folders = glob.glob(pathname=str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []

    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(path=folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        folder_docs = loader.load()

        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    return documents

def create_chunks(documents: list[Document]) -> list[Document]:
    """
    Split all documents in chunks.

    Args:
        documents (list): List of documents
    Returns:
        list[Document]: Splited chunks of the documents
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents=documents)
    return chunks

def create_embeddings(chunks: list[Document]) -> Chroma:
    """
    Create embeddings of chunks and store in Chroma vector database.

    Args:
        chunks (list[Document]): Chunks of the documents
    Returns:
        Chroma: Chroma vector store
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_NAME)

    return vectorstore

def main():
    documents = fetch_documents()
    chunks = create_chunks(documents=documents)
    vectorstore = create_embeddings(chunks=chunks)

    # Get the details of the vectore database created for printing in command line
    collection = vectorstore._collection
    vector_count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)

    print(f"Ingestion Completed: There are {vector_count} vectors with {dimensions} dimensions in vector store")

if __name__ == "__main__":
    main()