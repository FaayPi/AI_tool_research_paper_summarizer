# server.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from summarization_chain import build_summarization_chain

app = FastAPI(title="LangChain Summarization API")

# ---------------------------
# Load API Keys from Environment Variables
# ---------------------------
# For FastAPI/server context, use environment variables
# These should be set when deploying (e.g., in .env file or Docker)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    print("⚠️ Warning: API keys not found in environment variables")
    print("   Set OPENAI_API_KEY and PINECONE_API_KEY as environment variables")
    print("   The app will try to load them when summarization is called")

# --- Step 1: Define input model ---
class PaperInput(BaseModel):
    study_field: str
    research_question: str
    formatted_sub_questions: str
    formatted_hypotheses: str
    paper_title: str
    paper_text: str

# --- Step 2: Build the chain once on startup ---
# Note: This will use lazy-loaded keys from summarization_chain
chain = build_summarization_chain()

# --- Step 3: Endpoint to summarize paper ---
@app.post("/summarize")
def summarize_paper(input_data: PaperInput):
    """
    Summarize a research paper based on input framework.
    
    Args:
        input_data: PaperInput with paper details and research framework
    
    Returns:
        dict with 'summary' key containing the generated summary
    """
    # Convert input_data to dict for LangChain
    input_dict = input_data.dict()
    
    # Invoke chain
    output = chain.invoke(input_dict)
    
    return {"summary": output}

# --- Health check endpoint ---
@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "LangChain Summarization API"}