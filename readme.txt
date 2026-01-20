RAG-Based Document Question Answering Chatbot
Overview

This project implements a Retrieval-Augmented Generation (RAG)–based chatbot that enables intelligent, document-driven question answering for educational purposes. The system allows users to upload PDF documents, converts their content into vector embeddings, stores them in a FAISS vector database, and retrieves relevant context to generate accurate responses using a lightweight language model.

The chatbot is built with a React frontend for user interaction and a Python-based backend that handles document processing, semantic search, and text generation.

Features

Upload and process PDF documents

Automatic text extraction and chunking

Tokenization using Hugging Face tokenizer

Vector embedding storage using FAISS

Semantic similarity search for relevant context

Context-aware response generation using Tiny LLaMA 1.1B

User-friendly React-based web interface

Designed for educational and research use cases

Architecture
              Backend   →  PDF Processing  
                                  ↓  
                         Text Chunking & Tokenization  
                                  ↓  
                         FAISS Vector Database storing
                                  ↓  
                         Retrieval + Tiny LLaMA 1.1B  
                                  ↓  
                         Generated Answer → Frontend

Technologies Used

Frontend: React.js

Backend: Python Flask 

NLP & ML: Hugging Face Transformers

Vector Database: FAISS

LLM: Tiny LLaMA 1.1B

Document Processing: PyPDF2 