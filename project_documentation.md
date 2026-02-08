# Project Documentation

## Overview
This document provides detailed technical documentation for the Research Paper Insight Generator using RAG.

## Architecture
- Streamlit frontend
- Local LLM (Ollama – LLaMA 3)
- FAISS vector database
- HuggingFace embeddings

## Data Flow
PDF → Text Extraction → Chunking → Embeddings → FAISS → Retrieval → LLM → Summary

## Design Decisions
- RAG chosen to reduce hallucinations
- Local LLM used to avoid API costs and rate limits
- FAISS selected for fast similarity search

## Limitations
- Scanned PDFs without text layer are unsupported
- Performance depends on system resources
