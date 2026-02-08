# 🧠 Research Paper Insight Generator using RAG

**AI System for Intelligent Academic Paper Summarization**

This project implements an AI-driven system to **analyze, retrieve, and summarize academic research papers** using a **Retrieval-Augmented Generation (RAG)** pipeline.

Instead of relying on naive summarization, the system grounds generated outputs in the most relevant sections of the paper, producing **accurate, structured, and context-aware summaries**.

---

## 🎯 Motivation

Academic research papers are often lengthy and dense. This project aims to simplify the reading process by building a system that:

- Efficiently processes long academic documents  
- Identifies the most relevant contextual sections  
- Produces structured, research-oriented summaries  
- Reduces hallucinations commonly observed in standalone LLM outputs  

---

## ✨ Key Features

- PDF-based academic paper ingestion  
- Robust text extraction and chunking  
- Semantic embedding generation  
- Vector similarity search using FAISS  
- Retrieval-Augmented Generation (RAG) pipeline  
- LLM-driven structured summary generation  
- Interactive web interface built with Streamlit  

---

## 🧠 System Design

### Workflow Breakdown

1. User uploads an academic paper (PDF)  
2. Text is extracted from the document  
3. Content is split into semantic chunks  
4. Chunks are converted into vector embeddings  
5. Embeddings are indexed in a FAISS vector store  
6. Relevant chunks are retrieved via similarity search  
7. Retrieved context is passed to the LLM  
8. The LLM generates a concise academic summary  

---

## 🔄 End-to-End Pipeline

PDF Upload  
→ Text Extraction  
→ Chunking  
→ Embedding Generation  
→ FAISS Vector Store  
→ Context Retrieval  
→ LLM  
→ Structured Summary  

---

## 🔍 Why Retrieval-Augmented Generation?

RAG enhances summarization quality by grounding generation in retrieved source content, ensuring:

- Improved factual accuracy  
- Reduced hallucinations  
- Stronger alignment with the original document  
- Context-aware and reliable summaries  

---

## 📄 Example Output Structure

- Paper Overview  
- Research Objective  
- Methodology  
- Key Results  
- Conclusions and Contributions  

---

## 🛠️ Technology Stack

### Frontend
- Streamlit  

### Backend / AI
- Python  
- Retrieval-Augmented Generation (RAG)  
- Local Large Language Model (via Ollama)  

### Vector Database
- FAISS  

### Development Tools
- VS Code  
- GitHub  

---

## ▶️ Running the Application

```bash
pip install -r requirements.txt
streamlit run app.py
