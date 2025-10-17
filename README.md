# AI Research Paper Summarizer

A Retrieval-Augmented Generation (RAG) pipeline that analyzes research papers based on custom research frameworks. The application extracts key information from PDFs, retrieves relevant content, and generates structured summaries guided by your research questions and hypotheses.

---

## 🌐 Access the Application Online

You can use the app directly by following this link:

🔗 **Live Streamlit App:** [https://ysls2dqiutadl8td5wjnr8.streamlit.app/](https://ysls2dqiutadl8td5wjnr8.streamlit.app/)

## Usage Example

1. Start the application and define your research framework (field of stufy, main research question, sub-questions, hypotheses)
2. Upload a PDF research paper
3. The system will:
   - Extract and chunk the paper
   - Create embeddings and store in Pinecone
   - Retrieve relevant sections matching your framework
   - Generate a structured summary with page citations
4. Download the summary or save it to your collection

---

## 🧠 Project Description

This project implements an intelligent research paper analysis system combining:

- **PDF Processing**: Extracts text and metadata from research papers with page tracking  
- **Vector Embeddings**: Converts document chunks into semantic embeddings using OpenAI  
- **Vector Storage**: Stores embeddings in Pinecone for efficient similarity search  
- **RAG Pipeline**: Retrieves relevant document sections based on research questions  
- **LLM Summarization**: Generates comprehensive summaries aligned with your research framework  
- **Web Interface**: Streamlit-based UI for uploading PDFs and managing summaries  

Users define their research context (study field, main question, sub-questions, hypotheses), and the system analyzes uploaded papers specifically against that framework — including page citations.

---

## 🧩 Model Details

### Language Models

- **Embedding Model**: OpenAI `text-embedding-3-small`  
  - Dimension: 1536  
  - Used for converting text chunks into semantic vectors  

- **Summarization Model**: OpenAI `gpt-4o`  
  - Temperature: 0.3 (for consistent, focused summaries)  
  - Used for analyzing papers and generating structured summaries  

### Processing Configuration

- **Chunk Size**: 800 characters  
- **Chunk Overlap**: 100 characters  
- **Retrieval Count (k)**: 5–10 chunks per query  
- **Vector Database**: Pinecone (serverless, AWS region: `us-east-1`)  
- **Similarity Metric**: Cosine distance  

### Key Dependencies

- `langchain` — LLM orchestration  
- `langchain-openai` — OpenAI API integration  
- `langchain-pinecone` — Pinecone vector store  
- `pdfplumber` — PDF text extraction with page tracking  
- `PyPDF2` — PDF metadata extraction  
- `pinecone-client` — Vector database operations  
- `streamlit` — Web interface framework  

---

## 📁 File Structure

AI_project_module6/
├── streamlit_app.py                                  # Streamlit web interface
├── preprocessing_data.py                             # PDF processing and Pinecone upload
├── rag_pipeline.py                                   # Main RAG orchestration
├── summarization_chain.py                            # LLM summarization logic
├── server.py                                         # FastAPI backend server for REST API
├── requirements.txt                                  # Python package dependencies
├── prompt_engineering_skills.md                      # Prompt selection and notes
├── presentation_AI_research_paper_summarizer.pptx    # Architecture overview + QR code to app
├── ai_env                                            # Environment details
└── sample_research_papers/                           # Sample PDFs for testing
   ├── 30-39.pdf
   └── ...
└── README.md                                         # This file

---

## Limitations

- **Page Number Accuracy**: Page tracking is based on text matching and may be imprecise for PDFs with complex layouts or graphics
- **Content Extraction**: Title, Author, Images, tables, and formatted content in PDFs may not be extracted correctly
- **Language Support**: Currently optimized for English-language papers
- **API Costs**: Uses paid OpenAI API calls; large documents or high query volumes will incur costs
- **Context Window**: Retrieves up to 50 chunks maximum due to LLM token limits; very large papers may not be fully analyzed
- **Namespace Isolation**: Each PDF creates a separate Pinecone namespace; cross-document retrieval is not supported

---

## Future Improvements

- **Multi-Language Support**: Add language detection and support for papers in multiple languages
- **Advanced PDF Handling**: Implement table and figure extraction with OCR capabilities
- **Cross-Document Analysis**: Enable comparison and synthesis across multiple papers
- **Annotation Tools**: Allow users to highlight, annotate, and refine summaries within the UI
- **Citation Management**: Integrate with citation managers (BibTeX, Zotero, Mendeley)
- **Performance Optimization**: Implement caching and indexing strategies to reduce API costs