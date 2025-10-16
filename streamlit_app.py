# ===============================================
# app.py - AI Research Paper Summarizer
# ===============================================

import streamlit as st
import tempfile
import sys
import os

# Add project directory to path
sys.path.append("/Users/feepieper/Desktop/AI_project_module6")

from rag_pipeline import run_rag_pipeline  
from preprocessing_data import PINECONE_API_KEY

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="AI Research Paper Summarizer", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">AI Research Paper Summarizer</p>', unsafe_allow_html=True)

# ---------------------------
# Session State Initialization
# ---------------------------
if "master_inputs" not in st.session_state:
    st.session_state["master_inputs"] = {
        "study_field": "",
        "research_question": "",
        "sub_questions": [],
        "hypotheses": []
    }

if "summaries" not in st.session_state:
    st.session_state["summaries"] = []

if "pdf_current_output" not in st.session_state:
    st.session_state["pdf_current_output"] = ""

if "current_pdf_title" not in st.session_state:
    st.session_state["current_pdf_title"] = ""

if "processing" not in st.session_state:
    st.session_state["processing"] = False

# ---------------------------
# Tabs: Home and Summaries
# ---------------------------
tab_home, tab_summaries = st.tabs(["📝 Home", "📚 Collection"])

# ============================================================
# HOME TAB
# ============================================================
with tab_home:
    st.header("📋 Your Research Questions")
    st.markdown("Define your research framework that will guide the paper analysis.")
    
    # ---------------------------
    # Master Inputs Form
    # ---------------------------
    with st.form("master_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            study_field = st.text_input(
                "Field of Study",
                value=st.session_state["master_inputs"].get("study_field", ""),
                placeholder="e.g., Computer Science, Psychology, Biology",
                help="The academic field your research belongs to"
            )
            
            research_question = st.text_area(
                "Main Research Question",
                value=st.session_state["master_inputs"].get("research_question", ""),
                placeholder="e.g., How does AI impact educational outcomes?",
                help="Your overarching research question",
                height=100
            )
        
        with col2:
            sub_questions_input = st.text_area(
                "Sub-Questions (one per line)",
                value="\n".join(st.session_state["master_inputs"].get("sub_questions", [])),
                placeholder="e.g.,\nWhat are the main findings?\nWhat methodology was used?\nWhat are the limitations?",
                help="Specific questions to ask about each paper",
                height=100
            )
            
            hypotheses_input = st.text_area(
                "Hypotheses (one per line)",
                value="\n".join(st.session_state["master_inputs"].get("hypotheses", [])),
                placeholder="e.g.,\nAI improves learning outcomes\nAI reduces teacher workload",
                help="Your research hypotheses (optional)",
                height=100
            )
        
        submitted_master = st.form_submit_button("Save Inputs", use_container_width=True)

    if submitted_master:
        # Parse and save inputs
        parsed_sub_questions = [q.strip() for q in sub_questions_input.splitlines() if q.strip()]
        parsed_hypotheses = [h.strip() for h in hypotheses_input.splitlines() if h.strip()]
        
        # Validation
        if not study_field:
            st.warning("⚠️ Please provide a field of study.")
        elif not parsed_sub_questions:
            st.warning("⚠️ Please provide at least one sub-question.")
        else:
            st.session_state["master_inputs"] = {
                "study_field": study_field,
                "research_question": research_question,
                "sub_questions": parsed_sub_questions,
                "hypotheses": parsed_hypotheses
            }
            st.success("✅ Master inputs saved successfully!")

    # ---------------------------
    # Display Current Master Inputs
    # ---------------------------
    if st.session_state["master_inputs"].get("sub_questions"):
        st.divider()
        st.subheader("📊 Current Master Inputs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📖 Research Framework", expanded=True):
                st.markdown(f"**Field of Study:**  \n{st.session_state['master_inputs']['study_field']}")
                st.markdown(f"**Main Research Question:**  \n{st.session_state['master_inputs']['research_question']}")
        
        with col2:
            with st.expander(f"❓ Sub-Questions ({len(st.session_state['master_inputs']['sub_questions'])})", expanded=True):
                for idx, q in enumerate(st.session_state['master_inputs']['sub_questions'], 1):
                    st.markdown(f"{idx}. {q}")
            
            if st.session_state['master_inputs']['hypotheses']:
                with st.expander(f"💡 Hypotheses ({len(st.session_state['master_inputs']['hypotheses'])})", expanded=False):
                    for idx, h in enumerate(st.session_state['master_inputs']['hypotheses'], 1):
                        st.markdown(f"{idx}. {h}")

    st.divider()

    # ---------------------------
    # PDF Upload + Processing Section
    # ---------------------------
    st.header("📄 Upload & Analyze Research Paper")
    
    # Check if master inputs are defined
    if not st.session_state["master_inputs"].get("sub_questions"):
        st.warning("⚠️ Please define Master Inputs first before uploading a PDF.")
    else:
        with st.form("pdf_form"):
            pdf_file = st.file_uploader(
                "Upload PDF", 
                type="pdf", 
                key="pdf_file",
                help="Upload a research paper in PDF format"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                pdf_title = st.text_input(
                    "Paper Title (optional)", 
                    key="pdf_title",
                    placeholder="Leave empty to extract from PDF"
                )
            with col2:
                pdf_author = st.text_input(
                    "Author (optional)", 
                    key="pdf_author",
                    placeholder="e.g., John Doe et al."
                )
            
            submit_pdf = st.form_submit_button(
                "Process & Summarize Research Paper", 
                use_container_width=True,
                disabled=st.session_state.get("processing", False)
            )

        if submit_pdf:
            if not pdf_file:
                st.warning("⚠️ Please upload a PDF file.")
            else:
                # Set processing state
                st.session_state["processing"] = True
                
                # Save PDF to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.read())
                    tmp_pdf_path = tmp_file.name

                # Progress indicator
                progress_text = "🔄 Running ... This may take 1-2 minutes."
                progress_bar = st.progress(0, text=progress_text)
                
                try:
                    # Update progress
                    progress_bar.progress(20, text="📄 Extracting PDF text...")
                    
                    # Run the full RAG pipeline
                    summary_output = run_rag_pipeline(
                        pdf_path=tmp_pdf_path,
                        pinecone_api_key=PINECONE_API_KEY,
                        master_inputs=st.session_state["master_inputs"],
                        paper_title=pdf_title or pdf_file.name
                    )
                    
                    progress_bar.progress(100, text="✅ Complete!")

                    # Store results
                    st.session_state["pdf_current_output"] = summary_output["summary"]
                    st.session_state["current_pdf_title"] = pdf_title or pdf_file.name

                    # Add to summaries list
                    st.session_state["summaries"].append({
                        "title": pdf_title or pdf_file.name,
                        "author": pdf_author or "Unknown",
                        "summary": summary_output["summary"],
                        "metadata": summary_output.get("metadata", {})
                    })

                    # Clean up temp file
                    try:
                        os.remove(tmp_pdf_path)
                    except:
                        pass

                    st.success("✅ Summary generated!")

                except Exception as e:
                    st.error(f"❌ Error while processing PDF: {e}")
                    
                    # Show detailed error in expander
                    with st.expander("🔍 Show detailed error"):
                        import traceback
                        st.code(traceback.format_exc())
                    
                    # Clean up temp file
                    try:
                        os.remove(tmp_pdf_path)
                    except:
                        pass
                
                finally:
                    # Reset processing state
                    st.session_state["processing"] = False
                    progress_bar.empty()

    # ---------------------------
    # Show Current PDF Output
    # ---------------------------
    if st.session_state.get("pdf_current_output"):
        st.divider()
        st.subheader(f"📊 Summary: {st.session_state.get('current_pdf_title', 'Uploaded PDF')}")
        
        # Parse and display summary sections
        summary_text = st.session_state["pdf_current_output"]
        
        # Try to split by headers (###)
        if "###" in summary_text:
            sections = summary_text.split("###")
            for section in sections:
                if section.strip():
                    lines = section.strip().split("\n", 1)
                    if len(lines) > 1:
                        question = lines[0].strip()
                        answer = lines[1].strip()
                        with st.expander(f"{question}", expanded=True):
                            st.markdown(answer)
                    else:
                        st.markdown(section.strip())
        else:
            # If no headers, just show the full text
            st.markdown(summary_text)
        
        # Download button
        st.download_button(
            label="Download",
            data=summary_text,
            file_name=f"summary_{st.session_state.get('current_pdf_title', 'paper')}.txt",
            mime="text/plain"
        )

# ============================================================
# SUMMARIES TAB
# ============================================================
with tab_summaries:
    st.header("All Saved Summaries")
    
    if st.session_state["summaries"]:
        # Summary counter and controls
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Total summaries: **{len(st.session_state['summaries'])}**")
        with col2:
            if st.button("Clear All", use_container_width=True):
                st.session_state["summaries"] = []
                st.session_state["pdf_current_output"] = ""
                # Clear all edit mode states
                keys_to_delete = [k for k in st.session_state.keys() if k.startswith("edit_mode_")]
                for key in keys_to_delete:
                    del st.session_state[key]
                st.rerun()
        
        st.divider()
        
        # Display summaries in reverse chronological order
        for idx, item in enumerate(reversed(st.session_state["summaries"]), start=1):
            actual_idx = len(st.session_state["summaries"]) - idx  # real index in list
            
            # Initialize edit mode state for this item
            edit_key = f"edit_mode_{actual_idx}"
            if edit_key not in st.session_state:
                st.session_state[edit_key] = False

            with st.container():
                # Main row: Title/Author display or edit + controls
                col1, col2 = st.columns([8, 1])
                
                with col1:
                    if st.session_state[edit_key]:
                        # EDIT MODE: Show input fields with Save/Cancel buttons
                        input_col, buttons_col = st.columns([6, 2])
                        with input_col:
                            new_title = st.text_input(
                                "Title", 
                                value=item.get("title", ""), 
                                key=f"title_input_{idx}",
                                label_visibility="collapsed",
                                placeholder="Enter title"
                            )
                            new_author = st.text_input(
                                "Author", 
                                value=item.get("author", ""), 
                                key=f"author_input_{idx}",
                                label_visibility="collapsed",
                                placeholder="Enter author"
                            )
                        with buttons_col:
                            if st.button("💾 Save", key=f"save_{idx}", help="Save changes", use_container_width=True):
                                # Save the changes
                                st.session_state["summaries"][actual_idx]["title"] = new_title
                                st.session_state["summaries"][actual_idx]["author"] = new_author
                                st.session_state[edit_key] = False
                                st.success("✅ Saved!")
                                st.rerun()
                            if st.button("❌ Cancel", key=f"cancel_{idx}", help="Cancel", use_container_width=True):
                                st.session_state[edit_key] = False
                                st.rerun()
                    else:
                        # DISPLAY MODE: Show as text
                        st.markdown(f"**Title:** {item.get('title', 'Untitled')}")
                        st.markdown(f"**Author:** {item.get('author', 'Unknown')}")
                
                with col2:
                    # Edit and Delete buttons stacked vertically
                    if st.button("✏️ Edit", key=f"edit_btn_{idx}", help="Edit title and author", use_container_width=True):
                        st.session_state[edit_key] = True
                        st.rerun()
                    
                    if st.button("🗑️ Delete", key=f"delete_{idx}", help="Delete summary", use_container_width=True):
                        st.session_state["summaries"].pop(actual_idx)
                        # Clean up edit mode state
                        if edit_key in st.session_state:
                            del st.session_state[edit_key]
                        st.rerun()

                # Summary content
                summary_text = item["summary"]

                # Parse and display summary sections
                if "###" in summary_text:
                    sections = summary_text.split("###")
                    for section in sections:
                        if section.strip():
                            lines = section.strip().split("\n", 1)
                            if len(lines) > 1:
                                question = lines[0].strip()
                                answer = lines[1].strip()
                                with st.expander(f"{question}"):
                                    st.markdown(answer)
                            else:
                                st.markdown(section.strip())
                else:
                    with st.expander("📄 Full Summary", expanded=False):
                        st.markdown(summary_text)

                # Download button
                st.download_button(
                    label="Download",
                    data=summary_text,
                    file_name=f"summary_{item.get('title', 'paper')}.txt",
                    mime="text/plain",
                    key=f"download_{idx}"
                )

                st.divider()

    else:
        # Empty state
        st.info("📭 No summaries generated yet.")
        st.markdown("""
        ### 🚀 Get Started:
        1. Go to the **Home** tab  
        2. Define your **Master Research Inputs**  
        3. Upload a **PDF research paper**  
        4. Click **Process & Summarize**  
        5. Your summary will appear here!
        """)


# ---------------------------
# Sidebar (Optional Info)
# ---------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a **RAG (Retrieval-Augmented Generation)** pipeline to:
    
    1. 📄 Extract text from PDFs
    2. 🔪 Chunk the content
    3. 🧮 Create embeddings
    4. 📊 Store in Pinecone
    5. 🔍 Retrieve relevant passages
    6. 🤖 Generate summaries with GPT-4
    
    ---
    
    **Technologies:**
    - LangChain
    - OpenAI GPT-4o
    - Pinecone Vector DB
    - Streamlit
    """)
    
    st.divider()
    
    st.header("📊 Session Stats")
    st.metric("Master Inputs Defined", "✅" if st.session_state["master_inputs"].get("sub_questions") else "❌")
    st.metric("Total Summaries", len(st.session_state["summaries"]))
    
    if st.session_state["master_inputs"].get("sub_questions"):
        st.metric("Sub-Questions", len(st.session_state["master_inputs"]["sub_questions"]))