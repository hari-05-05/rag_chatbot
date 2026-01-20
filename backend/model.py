"""
RAG-based Educational PDF Chatbot using Hugging Face Models
Complete implementation with persistence and educational optimization
"""

import os
import pickle
import json
import warnings
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

# PDF processing
import PyPDF2
from PyPDF2 import PdfReader

# Text processing and splitting
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

# Vector storage
import faiss

# Text generation
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    GenerationConfig
)
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

class PDFProcessor:
    """Handles PDF text extraction with educational content optimization"""

    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""

                print(f"Extracting text from {len(pdf_reader.pages)} pages...")

                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text
                        text += "\n"

                return text.strip()

        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        # Remove excessive whitespace and newlines
        import re

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)

        # Remove page markers for cleaner chunking
        text = re.sub(r'--- Page \d+ ---', '', text)

        # Remove excessive newlines
        text = re.sub(r'\n+', '\n', text)

        return text.strip()

class EducationalTextSplitter:
    """Optimized text splitting for educational content"""

    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        """
        Initialize with parameters optimized for educational content

        Args:
            chunk_size: Target size of each chunk (in characters)
            overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK"""
        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception:
            # Fallback to simple splitting
            sentences = text.split('. ')
            return [s.strip() + '.' for s in sentences if s.strip()]

    def create_chunks(self, text: str) -> List[str]:
        """
        Create meaningful chunks optimized for educational content
        """
        sentences = self.split_by_sentences(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                    # Start new chunk with overlap from previous chunk
                    if self.overlap > 0 and chunks:
                        overlap_text = current_chunk[-self.overlap:].strip()
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    # Single sentence is longer than chunk_size
                    chunks.append(sentence)
                    current_chunk = ""
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # Add remaining text
        if current_chunk:
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if len(chunk.strip()) > 20]  # Filter very short chunks

class RAGVectorStore:
    """Vector storage and retrieval using FAISS"""

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with embedding model"""
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
        self.chunks = []
        self.chunk_metadata = []
        self.embedding_model_name = embedding_model_name


    def add_chunks(self, chunks: List[str], metadata: Optional[List[Dict]] = None):
        """Add chunks to the vector store"""
        print(f"Generating embeddings for {len(chunks)} chunks...")

        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))

        # Store chunks and metadata
        self.chunks.extend(chunks)
        if metadata:
            self.chunk_metadata.extend(metadata)
        else:
            self.chunk_metadata.extend([{"chunk_id": i} for i in range(len(chunks))])

        print(f"Added {len(chunks)} chunks to vector store")

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score),
                    "metadata": self.chunk_metadata[idx]
                })

        return results

    def save(self, save_dir: str):
        """Save the vector store to disk"""
        os.makedirs(save_dir, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_dir, "faiss_index.index"))

        # Save chunks and metadata
        with open(os.path.join(save_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

        with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.chunk_metadata, f)

        # Save configuration
        config = {
            "embedding_model_name": self.embedding_model_name,
            "embedding_dim": self.embedding_dim,
            "num_chunks": len(self.chunks),
            "created_at": datetime.now().isoformat()
        }

        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"Vector store saved to {save_dir}")

    @classmethod
    def load(cls, save_dir: str):
        """Load the vector store from disk"""
        # Load configuration
        with open(os.path.join(save_dir, "config.json"), "r") as f:
            config = json.load(f)

        # Initialize instance
        instance = cls(config["embedding_model_name"])

        # Load FAISS index
        instance.index = faiss.read_index(os.path.join(save_dir, "faiss_index.index"))

        # Load chunks and metadata
        with open(os.path.join(save_dir, "chunks.pkl"), "rb") as f:
            instance.chunks = pickle.load(f)

        with open(os.path.join(save_dir, "metadata.pkl"), "rb") as f:
            instance.chunk_metadata = pickle.load(f)

        print(f"Vector store loaded from {save_dir}")
        print(f"Loaded {len(instance.chunks)} chunks")

        return instance

class EducationalTextGenerator:
    """Text generation optimized for educational responses"""

    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        """Initialize with a lightweight text generation model"""
        print(f"Loading text generation model: {model_name}")

        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print("Text generation model loaded successfully")

        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Falling back to text generation pipeline...")

            # Fallback to pipeline
            self.pipeline = pipeline(
                "text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            self.model = None
            self.tokenizer = None

    def generate_response(self, context: str, question: str, max_length: int = 200) -> str:
        """Generate educational response based on context and question"""

        # Create educational prompt
        prompt = f"""Based on the following educational content, please provide a clear and informative answer to the question.

Educational Content:
{context}

Question: {question}

Answer:"""

        try:
            if self.model and self.tokenizer:
                # Use model directly
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)

                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=inputs.shape[1] + max_length,
                        num_return_sequences=1,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        top_p=0.9
                    )

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract only the generated part
                response = response[len(prompt):].strip()

            else:
                # Use pipeline
                outputs = self.pipeline(
                    prompt,
                    max_length=len(prompt.split()) + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
                response = outputs[0]['generated_text'][len(prompt):].strip()

            return response if response else "I'm sorry, I couldn't generate a proper response based on the provided context."

        except Exception as e:
            print(f"Error in text generation: {e}")
            return f"Based on the provided content: {context[:200]}... I can help answer your question about {question}, but I encountered a technical issue generating the full response."

class EducationalRAGChatbot:
    """Main RAG chatbot class for educational content"""

    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 generation_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize the RAG chatbot"""

        self.pdf_processor = PDFProcessor()
        self.text_splitter = EducationalTextSplitter(chunk_size=400, overlap=50)
        self.vector_store = None
        self.text_generator = EducationalTextGenerator(generation_model)

        print("Educational RAG Chatbot initialized!")

    def load_pdf(self, pdf_path: str):
        """Load and process PDF file"""
        print(f"Loading PDF: {pdf_path}")

        # Extract text
        raw_text = self.pdf_processor.extract_text_from_pdf(pdf_path)
        cleaned_text = self.pdf_processor.clean_text(raw_text)

        # Create chunks
        chunks = self.text_splitter.create_chunks(cleaned_text)

        print(f"Created {len(chunks)} chunks from PDF")

        # Initialize vector store if not exists
        if self.vector_store is None:
            self.vector_store = RAGVectorStore()

        # Add chunks to vector store
        self.vector_store.add_chunks(chunks)

        return len(chunks)

    def save_knowledge_base(self, save_dir: str = "educational_rag_kb"):
        """Save the knowledge base for future use"""
        if self.vector_store is None:
            raise ValueError("No knowledge base to save. Please load a PDF first.")

        self.vector_store.save(save_dir)

    def load_knowledge_base(self, save_dir: str = "educational_rag_kb"):
        """Load existing knowledge base"""
        self.vector_store = RAGVectorStore.load(save_dir)

    def answer_question(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Answer a question using RAG"""
        if self.vector_store is None:
            return {
                "answer": "Please load a PDF or knowledge base first.",
                "sources": [],
                "confidence": "low"
            }

        # Retrieve relevant chunks
        results = self.vector_store.search(question, k=k)

        if not results:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "confidence": "low"
            }

        # Combine context from top results
        context = "\n\n".join([result["chunk"] for result in results])

        # Generate answer
        answer = self.text_generator.generate_response(context, question)

        # Determine confidence based on similarity scores
        avg_score = sum([r["score"] for r in results]) / len(results)
        confidence = "high" if avg_score > 0.7 else "medium" if avg_score > 0.5 else "low"

        return {
            "answer": answer,
            "sources": results,
            "confidence": confidence,
            "context_used": context[:200] + "..." if len(context) > 200 else context
        }

    def interactive_chat(self):
        """Start interactive chat session"""
        print("\n" + "="*60)
        print("   Educational RAG Chatbot - Interactive Mode")
        print("="*60)
        print("Ask questions about your PDF documents!")
        print("Commands: 'quit' to exit, 'save' to save knowledge base")
        print("="*60 + "\n")

        while True:
            try:
                question = input("\nYou: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using Educational RAG Chatbot!")
                    break

                if question.lower() == 'save':
                    try:
                        self.save_knowledge_base()
                        print("Knowledge base saved successfully!")
                    except Exception as e:
                        print(f"Error saving knowledge base: {e}")
                    continue

                if not question:
                    continue

                print("\nChatbot: Thinking...")
                response = self.answer_question(question)

                print(f"\nChatbot: {response['answer']}")
                print(f"\nConfidence: {response['confidence']}")

                if response['sources']:
                    print(f"\nSources used ({len(response['sources'])} chunks):")
                    for i, source in enumerate(response['sources'][:2], 1):
                        preview = source['chunk'][:100] + "..." if len(source['chunk']) > 100 else source['chunk']
                        print(f"  {i}. [Score: {source['score']:.3f}] {preview}")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")

# Example usage and demonstration
def main():
    """Main function demonstrating the RAG chatbot"""

    print("\n" + "="*60)
    print("      Educational RAG Chatbot with Hugging Face")
    print("="*60)

    # Initialize chatbot
    chatbot = EducationalRAGChatbot()

    # Check if saved knowledge base exists
    kb_dir = "educational_rag_kb"
    if os.path.exists(kb_dir) and os.path.exists(os.path.join(kb_dir, "config.json")):
        print("\nFound existing knowledge base. Loading...")
        try:
            chatbot.load_knowledge_base(kb_dir)
            print("✓ Knowledge base loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading knowledge base: {e}")
            print("You'll need to load a new PDF.")

    # Interactive menu
    while True:
        print("\n" + "-"*40)
        print("Choose an option:")
        print("1. Load PDF file")
        print("2. Start chat session")
        print("3. Ask single question")
        print("4. Save knowledge base")
        print("5. Exit")
        print("-"*40)

        choice = input("Your choice (1-5): ").strip()

        if choice == '1':
            pdf_path = input("Enter PDF file path: ").strip()
            if os.path.exists(pdf_path):
                try:
                    num_chunks = chatbot.load_pdf(pdf_path)
                    print(f"✓ Successfully loaded PDF with {num_chunks} chunks!")
                except Exception as e:
                    print(f"✗ Error loading PDF: {e}")
            else:
                print("✗ PDF file not found!")

        elif choice == '2':
            if chatbot.vector_store is None:
                print("✗ Please load a PDF first!")
            else:
                chatbot.interactive_chat()

        elif choice == '3':
            if chatbot.vector_store is None:
                print("✗ Please load a PDF first!")
            else:
                question = input("Enter your question: ").strip()
                if question:
                    response = chatbot.answer_question(question)
                    print(f"\nAnswer: {response['answer']}")
                    print(f"Confidence: {response['confidence']}")

        elif choice == '4':
            if chatbot.vector_store is None:
                print("✗ No knowledge base to save!")
            else:
                try:
                    chatbot.save_knowledge_base()
                    print("✓ Knowledge base saved successfully!")
                except Exception as e:
                    print(f"✗ Error saving: {e}")

        elif choice == '5':
            print("\nThank you for using Educational RAG Chatbot!")
            break

        else:
            print("✗ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()