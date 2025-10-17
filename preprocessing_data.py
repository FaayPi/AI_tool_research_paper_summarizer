# ===============================================
# preprocessing_data.py - RAG Pipeline (Production Version)
# ===============================================

import os
import time
import uuid
from typing import Dict, List, Optional, Tuple
from PyPDF2 import PdfReader
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# ---------------------------
# Load Keys
# ---------------------------
import streamlit as st
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
except KeyError as e:
    raise ValueError(f"❌ API Key fehlt in Streamlit Secrets: {e}")

# ---------------------------
# Configuration constants
# ---------------------------
DEFAULT_INDEX_NAME = "research-papers"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 10
EMBED_MODEL = "text-embedding-3-small"

# ---------------------------
# PDF Extraction
# ---------------------------
def extract_pdf_metadata(pdf_path: str) -> Dict[str, str]:
    """Extract metadata from PDF file."""
    try:
        reader = PdfReader(pdf_path)
        meta = reader.metadata or {}
        return {
            "title": meta.get("/Title", "Unknown"),
            "author": meta.get("/Author", "Unknown"),
            "journal": meta.get("/Subject", "Unknown"),
            "source": os.path.basename(pdf_path)
        }
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {
            "title": "Unknown",
            "author": "Unknown",
            "journal": "Unknown",
            "source": os.path.basename(pdf_path)
        }

def extract_pdf_text(pdf_path: str) -> Tuple[str, Dict[int, str]]:
    """
    Extract all text from PDF with page mapping.
    
    Returns:
        Tuple of (full_text, page_mapping)
        - full_text: Complete text from all pages
        - page_mapping: Dict mapping page_number -> page_text
    """
    try:
        text_parts = []
        page_mapping = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    text_parts.append(text)
                    page_mapping[page_num] = text
                else:
                    page_mapping[page_num] = ""
        
        full_text = "\n\n".join(text_parts)
        return full_text, page_mapping
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return "", {}

# ---------------------------
# Text Chunking with Page Tracking
# ---------------------------
def chunk_text_with_pages(
    text: str,
    page_mapping: Dict[int, str],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Tuple[str, List[int]]]:
    """
    Split text into chunks and track which pages each chunk comes from.
    
    Returns:
        List of tuples: (chunk_text, list_of_page_numbers)
    """
    if not text or not page_mapping:
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    chunks_with_pages = []
    
    for chunk in chunks:
        pages_in_chunk = set()
        
        for page_num, page_text in page_mapping.items():
            if chunk in page_text or any(
                sub_chunk in page_text for sub_chunk in chunk.split("\n\n")[:1]
            ):
                pages_in_chunk.add(page_num)
        
        if not pages_in_chunk:
            pages_in_chunk = set(page_mapping.keys())
        
        chunks_with_pages.append((chunk, sorted(list(pages_in_chunk))))
    
    return chunks_with_pages

# ---------------------------
# Embeddings
# ---------------------------
def create_embeddings_model(model_name: str = EMBED_MODEL):
    """Initialize OpenAI embeddings model."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing!")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    return OpenAIEmbeddings(model=model_name)

# ---------------------------
# Pinecone Initialization
# ---------------------------
def init_pinecone(api_key: str, index_name: str = DEFAULT_INDEX_NAME, environment: str = "us-east-1"):
    """Initialize Pinecone and create index if needed."""
    pc = Pinecone(api_key=api_key)
    
    existing_indexes = pc.list_indexes().names()
    
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=environment)
        )
        time.sleep(10)
    
    index = pc.Index(index_name)
    return pc, index

# ---------------------------
# Vector Store
# ---------------------------
def create_vectorstore(
    chunks_with_pages: List[Tuple[str, List[int]]],
    metadata: Dict[str, str],
    embeddings,
    pinecone_index,
    namespace: str
):
    """Upload chunks to Pinecone vectorstore with page metadata."""
    docs = []
    for i, (chunk, pages) in enumerate(chunks_with_pages):
        pages_str = ",".join(str(p) for p in pages) if pages else "unknown"
        page_range = f"{min(pages)}-{max(pages)}" if pages else "unknown"
        
        doc_meta = {
            **metadata,
            "chunk_id": str(i),
            "page_numbers": pages_str,
            "page_range": page_range,
            "timestamp": str(time.time()),
            "text": chunk
        }
        docs.append(Document(page_content=chunk, metadata=doc_meta))

    vectorstore = PineconeVectorStore(
        index=pinecone_index,
        embedding=embeddings,
        text_key="text",
        namespace=namespace
    )
    
    vectorstore.add_documents(docs)
    return vectorstore

# ---------------------------
# Retriever Creation
# ---------------------------
def create_retriever(vectorstore, k: int = TOP_K, namespace: Optional[str] = None, filter_metadata: Optional[Dict] = None):
    """Create a retriever for similarity search."""
    search_kwargs = {"k": k}
    if namespace:
        search_kwargs["namespace"] = namespace
    if filter_metadata:
        search_kwargs["filter"] = filter_metadata

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    return retriever

# ---------------------------
# Cleanup Utility
# ---------------------------
def cleanup_namespace(index, namespace: str):
    """Delete all vectors in a namespace."""
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        print(f"Could not clear namespace: {e}")

# ---------------------------
# Complete PDF Preprocessing
# ---------------------------
def preprocess_pdf_complete(
    pdf_path: str,
    pinecone_api_key: str,
    index_name: str = DEFAULT_INDEX_NAME,
    namespace: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    k: int = TOP_K,
    cleanup_old_namespace: bool = False
):
    """
    Complete PDF preprocessing pipeline with page tracking.
    
    Args:
        pdf_path: Path to PDF file
        pinecone_api_key: Pinecone API key
        index_name: Name of Pinecone index
        namespace: Namespace for vectors (auto-generated if None)
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        k: Number of results for retriever
        cleanup_old_namespace: Whether to clean namespace before upload
    
    Returns:
        dict with keys: metadata, chunks_with_pages, vectorstore, retriever, namespace, index
    """
    namespace = namespace or f"paper-{uuid.uuid4().hex[:8]}"

    metadata = extract_pdf_metadata(pdf_path)
    text, page_mapping = extract_pdf_text(pdf_path)
    
    if not text:
        return {
            "metadata": metadata,
            "chunks_with_pages": [],
            "vectorstore": None,
            "retriever": None,
            "namespace": namespace,
            "index": None
        }

    chunks_with_pages = chunk_text_with_pages(
        text,
        page_mapping,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    if not chunks_with_pages:
        return {
            "metadata": metadata,
            "chunks_with_pages": [],
            "vectorstore": None,
            "retriever": None,
            "namespace": namespace,
            "index": None
        }

    embeddings = create_embeddings_model()
    pc, index = init_pinecone(pinecone_api_key, index_name)

    if cleanup_old_namespace:
        cleanup_namespace(index, namespace)

    vectorstore = create_vectorstore(chunks_with_pages, metadata, embeddings, index, namespace)

    expected_vectors = len(chunks_with_pages)
    max_wait_time = 60
    wait_interval = 5
    
    for i in range(max_wait_time // wait_interval):
        try:
            stats = index.describe_index_stats()
            ns_vectors = stats.namespaces.get(namespace, {}).get("vector_count", 0)
            
            if ns_vectors >= expected_vectors:
                break
            
            time.sleep(wait_interval)
        except Exception as e:
            time.sleep(wait_interval)

    retriever = create_retriever(
        vectorstore,
        k=k,
        namespace=namespace,
        filter_metadata={"source": metadata["source"]}
    )

    return {
        "metadata": metadata,
        "chunks_with_pages": chunks_with_pages,
        "vectorstore": vectorstore,
        "retriever": retriever,
        "namespace": namespace,
        "index": index
    }