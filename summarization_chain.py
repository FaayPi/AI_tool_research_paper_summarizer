# ===============================================
# summarization_chain.py - RAG Summarization Chain
# ===============================================

import sys
import os
sys.path.append('/Users/feepieper/Desktop/AI_project_module6')

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------
# Helper function to get API keys (lazy loading)
# ---------------------------
def get_openai_key():
    """
    Load OPENAI_API_KEY from Streamlit Secrets or environment variables.
    This is called at runtime, not at import time.
    """
    try:
        import streamlit as st
        try:
            return st.secrets["OPENAI_API_KEY"]
        except KeyError:
            # Fallback to environment variables
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY not found in Streamlit Secrets or environment variables")
            return key
    except ImportError:
        # If not in Streamlit context, use environment variables
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return key

# ---------------------------
# Build Summarization Chain
# ---------------------------
def build_summarization_chain(model_name="gpt-4o"):
    """
    Builds the LangChain summarization chain with system and human prompts.
    
    Args:
        model_name: OpenAI model to use (default: gpt-4o)
    
    Returns:
        LangChain chain (prompt | model | parser)
    """
    
    system_template = """
You are an academic research assistant specializing in {study_field}. Base your analysis **only on the content provided in the paper**. Do not add information from other sources unless explicitly requested.

Your task is to analyze and summarize research papers according to the following framework:

**Main Research Question:** {research_question}

**Sub-Questions:**
{formatted_sub_questions}

**Hypotheses:**
{formatted_hypotheses}

When given a research paper, you will:
1. Summarize the main findings clearly and concisely.
2. For each sub-question, extract relevant evidence from the paper. Cite sections or paragraphs, or provide a brief quote.
3. For each hypothesis, identify whether the evidence supports, partially supports, or contradicts it. Include citations or quotes.
4. Structure the output under headings for each sub-question and hypothesis.
5. If no evidence is present for a sub-question or hypothesis, explicitly state: "No evidence found in the paper."
6. Conclude with a brief overall assessment of how the paper addresses the main research question.
7. If no hypotheses are specified, disregard that step in your analysis.
8. **IMPORTANT: When citing page numbers, always use the format (Page X) or (Pages X-Y) at the end of citations. Use the page number information provided in the context.**

Provide a comprehensive yet focused analysis that directly addresses all aspects of the research framework.
"""
    
    human_template = """
**Paper Title:** {paper_title}

**Paper Content with Page Information:**
{paper_text_with_pages}

Please analyze this paper according to the research framework provided in the system prompt. Follow these instructions:

1. Summarize the main findings clearly and concisely.
2. For each sub-question, extract relevant evidence from the paper. Use your own summary style for approximately 70% of the text, and selectively include citations or brief quotes for about 30% of the content. Cite quotes correctly using quotation marks.
3. For each hypothesis, indicate whether the evidence supports, partially supports, or contradicts it. Again, summarize in your own style for the majority of the explanation, and include citations selectively.
4. **Include page numbers in the format (Page X) or (Pages X-Y) at the end of relevant citations. Use the page number information provided in the context above.**
5. Structure the output under headings for each sub-question and hypothesis, using bullet points.
6. If the paper does not provide evidence for a sub-question or hypothesis, explicitly state: "No evidence found in the paper."
7. Conclude with a brief overall assessment of how the paper addresses the main research question.
8. Maintain a balance between conciseness and completeness. Focus on clarity and relevance to the research framework.
"""
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])
    
    openai_key = get_openai_key()
    model = ChatOpenAI(model=model_name, temperature=0.3, api_key=openai_key)
    parser = StrOutputParser()
    
    return prompt | model | parser

# ---------------------------
# Format Content with Page Numbers
# ---------------------------
def format_content_with_pages(retrieved_docs):
    """
    Format retrieved documents with page number information.
    
    Args:
        retrieved_docs: List of Document objects with metadata
    
    Returns:
        str: Formatted text with page numbers included
    """
    formatted_sections = []
    
    for i, doc in enumerate(retrieved_docs, 1):
        page_range = doc.metadata.get("page_range", "unknown")
        page_info = f"[Pages: {page_range}]" if page_range != "unknown" else "[Pages: unknown]"
        
        formatted_sections.append(
            f"\n--- Document Chunk {i} {page_info} ---\n{doc.page_content}\n"
        )
    
    return "\n".join(formatted_sections)

# ---------------------------
# Run Summary Chain
# ---------------------------
def run_summary_chain(
    retriever, 
    master_inputs: dict, 
    paper_title: str = "Untitled Paper", 
    model_name: str = "gpt-4o"
):
    """
    Runs the summarization chain with retrieval and page number tracking.
    
    Args:
        retriever: LangChain retriever or NamespacedRetriever instance
        master_inputs: Dict containing:
            - study_field: Field of study
            - research_question: Main research question
            - sub_questions: List of sub-questions
            - hypotheses: List of hypotheses
        paper_title: Title of the paper
        model_name: OpenAI model to use
    
    Returns:
        str: Generated summary text with page numbers
    """
    
    if retriever is None:
        return "Error: No retriever available. Please check PDF preprocessing."
    
    # Build query from master inputs
    query_parts = []
    
    if master_inputs.get("research_question"):
        query_parts.append(master_inputs["research_question"])
    
    sub_questions = master_inputs.get("sub_questions", [])
    if sub_questions:
        query_parts.extend(sub_questions)
    
    hypotheses = master_inputs.get("hypotheses", [])
    if hypotheses:
        query_parts.extend(hypotheses)
    
    query = "\n".join(query_parts)
    
    # Retrieve documents
    try:
        retrieved_docs = retriever.invoke(query)
        
        if not retrieved_docs:
            return f"Unable to provide summary for '{paper_title}' - no relevant content was retrieved."
        
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return f"Error retrieving documents: {str(e)}"
    
    # Format content with page information
    paper_text_with_pages = format_content_with_pages(retrieved_docs)
    
    # Format master inputs
    formatted_sub_questions = "\n".join(
        f"- {q}" for q in sub_questions
    ) if sub_questions else "None provided"
    
    formatted_hypotheses = "\n".join(
        f"- {h}" for h in hypotheses
    ) if hypotheses else "None provided"
    
    input_data = {
        "study_field": master_inputs.get("study_field", "general research"),
        "research_question": master_inputs.get("research_question", "Not specified"),
        "formatted_sub_questions": formatted_sub_questions,
        "formatted_hypotheses": formatted_hypotheses,
        "paper_title": paper_title,
        "paper_text_with_pages": paper_text_with_pages,
    }
    
    # Run summarization chain
    try:
        chain = build_summarization_chain(model_name=model_name)
        summary = chain.invoke(input_data)
        return summary
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"Error generating summary: {str(e)}"

# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    print("This module should be imported and used via rag_pipeline.py")