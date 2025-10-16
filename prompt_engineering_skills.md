Prompt Engineering Skills - Examples 

This document presents a selection of my best prompts from my AI RAG project, in which I extensively used vibes coding. These prompts were used to design, debug, and refine my pipeline integrating Pinecone, LangChain, and Streamlit, demonstrating clear, iterative, and technically precise prompt engineering for AI research workflows.

🧩 Prompt 1
Original Prompt:
"please convert my langchain Pinecone code to the new syntax of langchain community"
Why It Was Effective:
It clearly specifies the task—updating code to match a new library API—while providing context about which library and module to focus on. This enabled precise code refactoring guidance.
Tag: RAG optimization / Library migration

🧩 Prompt 2
Original Prompt:
"Refactor the code so that it works with OpenAIEmbeddings"
Why It Was Effective:
It provides a concrete modification goal (switching embedding models) while keeping the context of the existing pipeline, ensuring technically correct refactoring guidance.
Tag: RAG optimization / Embeddings integration

🧩 Prompt 3
Original Prompt:
"without ANY adjustments. just so that I can copy the previous text in my notebook. I don’t want you to do any adjustments, just deliver for copy-paste ready! same for the preprocessing_data.py"
Why It Was Effective:
This prompt clearly constrained the model’s behavior, preventing unwanted improvements or formatting changes. It demonstrates precise control over AI output fidelity, crucial for transferring production-ready code.
Tag: Output control / Reproducibility

🧩 Prompt 5
Original Prompt:
"Why do I get the error code when I try to run my Streamlit app?"
Why It Was Effective:
It combines a real error scenario with a clear question, allowing the model to reason about stack traces and diagnose Python import issues.
Tag: Debugging / Streamlit integration

🧩 Prompt 6
Original Prompt:
"Explain how to test and validate that the retrieval component of the RAG pipeline is returning relevant and high-quality results. Suggest quantitative metrics or evaluation approaches."
Why It Was Effective:
Encourages analytical reasoning beyond implementation, guiding the model toward evaluation and metric design.
Tag: Evaluation & Testing Strategy

🧩 Prompt 7
Original Prompt:
"Adjust the prompt so that it would not always give citations for each subquestion and hypothesis but uses its own summary style and citation style on a 70:30 level."
Why It Was Effective:
Balances extractive (citation-based) and abstractive (summary-based) behavior, producing readable yet evidence-grounded research summaries.
Tag: Output style calibration / Summarization tuning

🧩 Prompt 8
Original Prompt:
"Add that every statement or bullet point should include the page or abstract in parentheses where the content can be found. If it cannot be accurately traced, no citation should be added. Do not invent anything."
Why It Was Effective:
Enforces conditional citation grounding, ensuring references are included only when verifiable.
Tag: Citation reliability / Factual grounding