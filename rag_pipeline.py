# ===============================================
# rag_pipeline.py - RAG Pipeline
# ===============================================

from preprocessing_data import preprocess_pdf_complete, PINECONE_API_KEY
from summarization_chain import run_summary_chain
from API_keys import OPENAI_API_KEY

# ---------------------------
# Configuration / Constants
# ---------------------------
DEFAULT_INDEX_NAME = "research-papers"
TOP_K = 5

# ---------------------------
# Custom Namespaced Retriever
# ---------------------------
class NamespacedRetriever:
    """
    Custom retriever that properly handles Pinecone namespaces.
    """
    
    def __init__(self, vectorstore, namespace, k, filter_dict=None):
        """
        Initialize the namespaced retriever.
        
        Args:
            vectorstore: PineconeVectorStore instance
            namespace: Pinecone namespace to search in
            k: Number of documents to retrieve
            filter_dict: Optional metadata filter
        """
        self.vectorstore = vectorstore
        self.namespace = namespace
        self.k = k
        self.filter_dict = filter_dict or {}
    
    def invoke(self, query):
        """
        Retrieve documents with namespace support.
        
        Args:
            query: Search query string
            
        Returns:
            List of Document objects
        """
        try:
            docs = self.vectorstore.similarity_search(
                query,
                k=self.k,
                namespace=self.namespace,
                filter=self.filter_dict
            )
            return docs
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
    
    def get_relevant_documents(self, query):
        """
        Alternative method name for compatibility with LangChain.
        """
        return self.invoke(query)

# ---------------------------
# Main RAG Pipeline
# ---------------------------
def run_rag_pipeline(
    pdf_path: str,
    master_inputs: dict,
    pinecone_api_key: str,
    paper_title: str = "Untitled Paper",
    index_name: str = DEFAULT_INDEX_NAME,
    top_k: int = TOP_K
):
    """
    Run the complete RAG pipeline.
    
    This function:
    1. Preprocesses the PDF (extract text, chunk, embed, upload to Pinecone)
    2. Creates a retriever that queries the Pinecone namespace
    3. Runs the summarization chain with all master inputs
    4. Returns the formatted summary
    
    Args:
        pdf_path: Path to PDF file
        master_inputs: Dict with:
            - 'sub_questions': List of sub-questions
            - 'research_question': Main research question
            - 'hypotheses': List of hypotheses
            - 'study_field': Field of study
        pinecone_api_key: Pinecone API key
        paper_title: Title of the paper
        index_name: Pinecone index name
        top_k: Number of chunks to retrieve per query
    
    Returns:
        dict with:
            - 'summary': Formatted summary text
            - 'details': Per sub-question breakdown
            - 'metadata': PDF metadata
            - 'namespace': Pinecone namespace used
    """
    
    # Preprocess PDF
    preprocess_result = preprocess_pdf_complete(
        pdf_path=pdf_path,
        pinecone_api_key=pinecone_api_key,
        index_name=index_name,
        namespace=None,
        k=top_k
    )
    
    vectorstore = preprocess_result["vectorstore"]
    namespace = preprocess_result["namespace"]
    metadata = preprocess_result["metadata"]
    
    if not vectorstore:
        return {
            "summary": "Error: Could not process PDF",
            "details": {},
            "metadata": metadata,
            "namespace": namespace
        }
    
    # Update paper title from metadata if not provided
    if paper_title == "Untitled Paper" and metadata["title"] != "Unknown":
        paper_title = metadata["title"]
    
    # Extract sub-questions
    sub_questions = master_inputs.get("sub_questions", [])
    if not sub_questions:
        return {
            "summary": "Error: No sub-questions provided",
            "details": {},
            "metadata": metadata,
            "namespace": namespace
        }
    
    # Create namespaced retriever
    retriever_k = top_k * max(len(sub_questions), 3)
    
    main_retriever = NamespacedRetriever(
        vectorstore=vectorstore,
        namespace=namespace,
        k=retriever_k,
        filter_dict={"source": metadata["source"]}
    )
    
    # Test retriever
    try:
        test_query = sub_questions[0] if sub_questions else "research methodology"
        test_docs = main_retriever.invoke(test_query)
        
        if not test_docs:
            return {
                "summary": "Error: No documents found in Pinecone",
                "details": {},
                "metadata": metadata,
                "namespace": namespace
            }
            
    except Exception as e:
        print(f"Retriever test failed: {e}")
        return {
            "summary": f"Error: Retriever failed - {str(e)}",
            "details": {},
            "metadata": metadata,
            "namespace": namespace
        }
    
    # Run summarization chain
    try:
        summary_text = run_summary_chain(
            retriever=main_retriever,
            master_inputs=master_inputs,
            paper_title=paper_title
        )
        
    except Exception as e:
        print(f"Summarization failed: {e}")
        return {
            "summary": f"Error during summarization: {str(e)}",
            "details": {},
            "metadata": metadata,
            "namespace": namespace
        }
    
    # Structure output
    sub_question_summaries = {}
    for sub_q in sub_questions:
        sub_question_summaries[sub_q] = {
            "summary": summary_text
        }
    
    return {
        "summary": summary_text,
        "details": sub_question_summaries,
        "metadata": metadata,
        "namespace": namespace
    }

# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    sample_inputs = {
        "study_field": "Computer Science",
        "research_question": "How does machine learning improve natural language processing?",
        "sub_questions": [
            "What is the main research question of this paper?",
            "What methodology was used in this study?",
            "What are the key findings and conclusions?"
        ],
        "hypotheses": [
            "Deep learning models outperform traditional methods",
            "Transfer learning improves model performance"
        ]
    }
    
    result = run_rag_pipeline(
        pdf_path="path/to/your/paper.pdf",
        master_inputs=sample_inputs,
        pinecone_api_key=PINECONE_API_KEY,
        paper_title="My Research Paper"
    )
    
    print("\nFINAL SUMMARY:")
    print("="*60)
    print(result["summary"])
    print("="*60)