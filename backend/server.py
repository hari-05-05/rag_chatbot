import os
import pickle
import json
import warnings
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import threading
from queue import Queue

# Flask imports
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# Core dependencies 
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

class SavedRAGVectorStore:
    """Lightweight vector store loader for accessing saved databases"""

    def __init__(self, save_dir: str):
        """Load existing vector store from directory"""
        self.save_dir = save_dir
        self.embedding_model = None
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.config = {}

        self._load_database()

    def _load_database(self):
        """Load all components of the saved database"""
        if not os.path.exists(self.save_dir):
            raise FileNotFoundError(f"Knowledge base directory not found: {self.save_dir}")

        # Load configuration
        config_path = os.path.join(self.save_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = json.load(f)

        print(f"Loading knowledge base from {self.save_dir}")
        print(f"Model: {self.config['embedding_model_name']}")
        print(f"Total chunks: {self.config['num_chunks']}")
        print(f"Created: {self.config['created_at']}")

        # Load embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.config["embedding_model_name"])

        # Load FAISS index
        index_path = os.path.join(self.save_dir, "faiss_index.index")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"✓ FAISS index loaded")
        else:
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        # Load chunks
        chunks_path = os.path.join(self.save_dir, "chunks.pkl")
        if os.path.exists(chunks_path):
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            print(f"✓ {len(self.chunks)} chunks loaded")
        else:
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        # Load metadata
        metadata_path = os.path.join(self.save_dir, "metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                self.chunk_metadata = pickle.load(f)
            print(f"✓ Chunk metadata loaded")
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        print("Knowledge base loaded successfully!")

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
                    "metadata": self.chunk_metadata[idx],
                    "chunk_id": idx
                })

        return results

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded database"""
        if not self.chunks:
            return {"error": "No data loaded"}

        chunk_lengths = [len(chunk) for chunk in self.chunks]
        return {
            "total_chunks": len(self.chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "embedding_model": self.config.get("embedding_model_name", "Unknown"),
            "embedding_dimension": self.config.get("embedding_dim", "Unknown"),
            "created_at": self.config.get("created_at", "Unknown")
        }

class ChatbotTextGenerator:
    """Optimized text generator for chatbot responses"""

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize with a conversational model"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading text generation model: {model_name}")
        print(f"Using device: {self.device}")

        try:
            # Use pipeline for simplicity and reliability - EXACTLY as in main model
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                max_length=512
            )
            print("✓ Text generation model loaded successfully")

        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Falling back to DialoGPT...")

            # Fallback to smaller model - EXACTLY as in main model
            self.generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",
                device_map="auto" if self.device == "cuda" else None
            )

    def generate_educational_response(self, context: str, question: str) -> str:
        """Generate educational response with improved prompting - IDENTICAL to main model"""

        # Create a more structured prompt - EXACTLY as in main model
        prompt = f"""You are an educational assistant. Based on the provided educational content, give a clear and informative answer.

Educational Content:
{context[:800]}  # Limit context to avoid token limits

Student Question: {question}

Educational Answer:"""

        try:
            # Generate response - EXACTLY as in main model
            response = self.generator(
                prompt,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )

            # Extract generated text - EXACTLY as in main model
            generated_text = response[0]['generated_text']
            answer = generated_text[len(prompt):].strip()

            # Clean up the answer - EXACTLY as in main model
            if answer:
                # Remove any incomplete sentences at the end
                sentences = answer.split('.')
                if len(sentences) > 1 and sentences[-1].strip() == '':
                    sentences = sentences[:-1]
                answer = '. '.join(sentences).strip()
                if answer and not answer.endswith('.'):
                    answer += '.'

            return answer if answer else "I can help explain this topic based on the content provided, but I need a bit more context to give you a complete answer."

        except Exception as e:
            print(f"Generation error: {e}")
            # Fallback response using context - EXACTLY as in main model
            return f"Based on the educational material: {context[:200]}... I can help answer questions about {question}. Could you ask a more specific question about this topic?"

    def generate_streaming_response(self, context: str, question: str):
        """Generate streaming response word by word for better UX"""
        try:
            # Get the full response first using the same method as non-streaming
            full_response = self.generate_educational_response(context, question)

            # Split into words and stream them
            words = full_response.split()
            for i, word in enumerate(words):
                if i == len(words) - 1:
                    yield word  # Last word without space
                else:
                    yield word + " "
                time.sleep(0.1)  # Small delay to simulate streaming

        except Exception as e:
            yield f"Error generating response: {str(e)}"

class EducationalChatbotAPI:
    """Flask API wrapper for the educational RAG chatbot - OPTIMIZED for consistency"""

    def __init__(self, knowledge_base_dir: str = "educational_rag_kb"):
        """Initialize the chatbot API"""
        self.kb_dir = knowledge_base_dir
        self.vector_store = None
        self.text_generator = None
        # Use thread-local storage for session history to avoid conflicts
        self._local = threading.local()

        print("\n" + "="*65)
        print("    Educational RAG Chatbot - API Server")
        print("="*65)

        self._load_components()

    @property
    def session_history(self):
        """Thread-safe session history"""
        if not hasattr(self._local, 'history'):
            self._local.history = []
        return self._local.history

    @session_history.setter
    def session_history(self, value):
        """Thread-safe session history setter"""
        self._local.history = value

    def _load_components(self):
        """Load vector store and text generator"""
        try:
            # Load vector store
            self.vector_store = SavedRAGVectorStore(self.kb_dir)

            # Initialize text generator
            self.text_generator = ChatbotTextGenerator()

            print("\n✓ All components loaded successfully!")

        except Exception as e:
            print(f"\n✗ Error loading components: {e}")
            print("Please ensure your knowledge base is properly saved.")
            self.vector_store = None
            self.text_generator = None

    def answer_question(self, question: str, num_sources: int = 3) -> Dict[str, Any]:
        """Answer question using the loaded knowledge base - IDENTICAL to main model"""
        if not self.vector_store or not self.text_generator:
            return {
                "answer": "Chatbot not properly initialized. Please check your knowledge base.",
                "sources": [],
                "confidence": "low",
                "error": "Components not loaded"
            }

        # Search for relevant information
        search_results = self.vector_store.search(question, k=num_sources)

        if not search_results:
            return {
                "answer": "I couldn't find relevant information in the knowledge base to answer your question. Try rephrasing or asking about a different topic.",
                "sources": [],
                "confidence": "low"
            }

        # Combine context from top results
        context_chunks = []
        for result in search_results:
            context_chunks.append(result["chunk"])

        combined_context = "\n\n".join(context_chunks)

        # Generate answer
        answer = self.text_generator.generate_educational_response(combined_context, question)

        # Calculate confidence based on similarity scores
        avg_score = sum([r["score"] for r in search_results]) / len(search_results)
        if avg_score > 0.75:
            confidence = "high"
        elif avg_score > 0.6:
            confidence = "medium"
        else:
            confidence = "low"

        # Store in session history (thread-safe)
        self.session_history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence
        })

        return {
            "answer": answer,
            "sources": search_results,
            "confidence": confidence,
            "context_preview": combined_context[:200] + "..." if len(combined_context) > 200 else combined_context
        }

    def get_streaming_response(self, question: str, num_sources: int = 3):
        """Get streaming response for real-time experience"""
        if not self.vector_store or not self.text_generator:
            yield "data: Error: Chatbot not properly initialized\n\n"
            return

        # Search for relevant information
        search_results = self.vector_store.search(question, k=num_sources)

        if not search_results:
            yield "data: I couldn't find relevant information to answer your question.\n\n"
            return

        # Combine context from top results
        context_chunks = []
        for result in search_results:
            context_chunks.append(result["chunk"])

        combined_context = "\n\n".join(context_chunks)

        # Stream the response
        for chunk in self.text_generator.generate_streaming_response(combined_context, question):
            yield f"data: {chunk}\n\n"

        # Signal completion
        yield "data: [DONE]\n\n"

    def get_database_info(self):
        """Get information about the loaded database"""
        if not self.vector_store:
            return {"error": "No database loaded"}

        return self.vector_store.get_database_stats()

    def clear_session_history(self):
        """Clear session history for current thread"""
        self.session_history = []

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global chatbot instance
chatbot = None
chatbot_lock = threading.Lock()  # Thread safety for initialization

def initialize_chatbot():
    """Initialize the chatbot on server startup"""
    global chatbot
    with chatbot_lock:
        if chatbot is None:
            try:
                chatbot = EducationalChatbotAPI("educational_rag_kb")
                print("Chatbot initialized successfully!")
                return True
            except Exception as e:
                print(f"Failed to initialize chatbot: {e}")
                return False
        return chatbot is not None

def get_chatbot():
    """Get chatbot instance with lazy initialization"""
    global chatbot
    if chatbot is None:
        initialize_chatbot()
    return chatbot

# API Routes

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    chatbot_instance = get_chatbot()
    return jsonify({
        "status": "running",
        "service": "Educational RAG Chatbot API",
        "timestamp": datetime.now().isoformat(),
        "chatbot_ready": chatbot_instance is not None and chatbot_instance.vector_store is not None
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint for asking questions"""
    try:
        # Get request data
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({
                "error": "No question provided",
                "required_format": {"question": "Your question here"}
            }), 400

        question = data['question'].strip()

        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        # Get chatbot instance
        chatbot_instance = get_chatbot()

        # Check if chatbot is ready
        if not chatbot_instance or not chatbot_instance.vector_store:
            return jsonify({
                "error": "Chatbot not initialized",
                "message": "Please check if the knowledge base is properly loaded"
            }), 503

        # Get response from chatbot
        response = chatbot_instance.answer_question(question)

        return jsonify({
            "success": True,
            "question": question,
            "answer": response["answer"],
            "confidence": response["confidence"],
            "sources_count": len(response["sources"]),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint for real-time responses"""
    try:
        # Get request data
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400

        question = data['question'].strip()

        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        # Get chatbot instance
        chatbot_instance = get_chatbot()

        # Check if chatbot is ready
        if not chatbot_instance or not chatbot_instance.vector_store:
            return jsonify({"error": "Chatbot not initialized"}), 503

        # Return streaming response
        return Response(
            stream_with_context(chatbot_instance.get_streaming_response(question)),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )

    except Exception as e:
        def error_stream():
            yield f"data: Error: {str(e)}\n\n"

        return Response(
            stream_with_context(error_stream()),
            mimetype='text/event-stream'
        )

@app.route('/api/database/info', methods=['GET'])
def database_info():
    """Get information about the knowledge base"""
    try:
        chatbot_instance = get_chatbot()
        if not chatbot_instance:
            return jsonify({"error": "Chatbot not initialized"}), 503

        info = chatbot_instance.get_database_info()
        return jsonify({
            "success": True,
            "database_info": info,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            "error": "Failed to get database info",
            "message": str(e)
        }), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get chat history"""
    try:
        chatbot_instance = get_chatbot()
        if not chatbot_instance:
            return jsonify({"error": "Chatbot not initialized"}), 503

        # Get last N entries (default 10)
        limit = request.args.get('limit', 10, type=int)
        history = chatbot_instance.session_history[-limit:] if limit > 0 else chatbot_instance.session_history

        return jsonify({
            "success": True,
            "history": history,
            "total_questions": len(chatbot_instance.session_history),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            "error": "Failed to get history",
            "message": str(e)
        }), 500

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear chat history"""
    try:
        chatbot_instance = get_chatbot()
        if not chatbot_instance:
            return jsonify({"error": "Chatbot not initialized"}), 503

        chatbot_instance.clear_session_history()

        return jsonify({
            "success": True,
            "message": "Chat history cleared",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            "error": "Failed to clear history",
            "message": str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET / - Health check",
            "POST /api/chat - Ask questions",
            "POST /api/chat/stream - Streaming responses",
            "GET /api/database/info - Database information",
            "GET /api/history - Get chat history",
            "POST /api/history/clear - Clear history"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong on the server"
    }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("    Educational RAG Chatbot - Flask API Server")
    print("="*60)
    print("\nInitializing server components...")

    # Initialize chatbot
    success = initialize_chatbot()

    if success and chatbot and chatbot.vector_store:
        print("\n✅ Server ready!")
        print("\nAvailable endpoints:")
        print("  • GET  /                    - Health check")
        print("  • POST /api/chat            - Ask questions (JSON)")
        print("  • POST /api/chat/stream     - Streaming responses (SSE)")
        print("  • GET  /api/database/info   - Database information")
        print("  • GET  /api/history         - Get chat history")
        print("  • POST /api/history/clear   - Clear history")
        print("\n" + "="*60)

        # Run Flask app
        app.run(
            host='0.0.0.0',  # Accept connections from any IP
            port=5000,       # Default Flask port
            debug=False,     # Disable debug mode for consistent behavior
            threaded=True    # Enable threading for concurrent requests
        )
    else:
        print("\n❌ Failed to initialize server!")
        print("Please check your knowledge base directory and try again.")