# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from summarization_chain import build_summarization_chain

app = FastAPI(title="LangChain Summarization API")

# --- Step 1: Define input model ---
class PaperInput(BaseModel):
    study_field: str
    research_question: str
    formatted_sub_questions: str
    formatted_hypotheses: str
    paper_title: str
    paper_text: str

# --- Step 2: Build the chain once on startup ---
chain = build_summarization_chain()

# --- Step 3: Endpoint to summarize paper ---
@app.post("/summarize")
def summarize_paper(input_data: PaperInput):
    # Convert input_data to dict for LangChain
    input_dict = input_data.dict()
    
    # Invoke chain
    output = chain.invoke(input_dict)
    
    return {"summary": output}
