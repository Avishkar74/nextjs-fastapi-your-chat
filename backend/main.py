from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our RAG system - try enhanced system first, then fall back to original
enhanced_rag_system = None
rag_system = None

try:
    from enhanced_rag_system import EnhancedRAGSystem
    logger.info("Enhanced RAG system available")
    ENHANCED_AVAILABLE = True
except Exception as e:
    logger.warning(f"Enhanced RAG system not available: {e}")
    ENHANCED_AVAILABLE = False

try:
    from rag_system import RAGSystem
    logger.info("Original RAG system available")
    ORIGINAL_AVAILABLE = True
except Exception as e:
    logger.warning(f"Original RAG system not available: {e}")
    ORIGINAL_AVAILABLE = False

# Global RAG system instances

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # Add session support
    
class IndexRequest(BaseModel):
    github_url: str
    
class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = []
    session_id: Optional[str] = None  # Return session ID
    conversation_length: Optional[int] = None  # Number of exchanges in conversation
    conversation_length: Optional[int] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global rag_system, enhanced_rag_system
    try:
        # Startup - Initialize both systems if available
        if ENHANCED_AVAILABLE:
            enhanced_rag_system = EnhancedRAGSystem()
            logger.info("Enhanced RAG system initialized successfully")
        
        if ORIGINAL_AVAILABLE:
            rag_system = RAGSystem()
            logger.info("Original RAG system initialized successfully")
            
        if not enhanced_rag_system and not rag_system:
            logger.error("No RAG system could be initialized")
    except Exception as e:
        logger.error(f"Failed to initialize RAG systems: {e}")
        # Don't raise error - let the app start and show helpful error messages
    
    yield
    
    # Shutdown
    logger.info("RAG Chat API shutting down")

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chat API",
    description="A RAG-powered chat API using Gemini 2.0 Flash and Supabase",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "https://*.vercel.app",   # Vercel deployments
        "*"  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "RAG Chat API is running!",
        "status": "healthy",
        "docs": "Visit /docs for API documentation"
    }

@app.get("/api/health")
async def health_check():
    """Extended health check"""
    return {
        "status": "healthy",
        "rag_system_initialized": rag_system is not None,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "gemini_key_set": bool(os.getenv("GEMINI_API_KEY")),
        "supabase_url_set": bool(os.getenv("SUPABASE_URL")),
        "supabase_key_set": bool(os.getenv("SUPABASE_KEY"))
    }

@app.post("/api/index")
async def index_repository(request: IndexRequest):
    """Index a GitHub repository for RAG search"""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        if not request.github_url:
            raise HTTPException(status_code=400, detail="GitHub URL is required")
        
        logger.info(f"Starting to index repository: {request.github_url}")
        
        # Index the repository
        result = await rag_system.index_github_repo(request.github_url)
        
        return {
            "message": "Repository indexed successfully",
            "github_url": request.github_url,
            "documents_processed": result.get("documents_processed", 0),
            "chunks_created": result.get("chunks_created", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error indexing repository: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to index repository: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint that uses RAG to answer questions"""
    try:
        # Temporarily use only the original system until enhanced is fixed
        if rag_system:
            logger.info(f"Processing chat request with original system: {request.message}")
            
            if not request.message or not request.message.strip():
                raise HTTPException(status_code=400, detail="Message cannot be empty")
            
            result = await rag_system.answer_question(request.message)
            
            return ChatResponse(
                answer=result["answer"],
                sources=result.get("sources", [])
            )
        else:
            raise HTTPException(status_code=500, detail="RAG system not available")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat request: {str(e)}")

@app.post("/api/enhanced-chat", response_model=ChatResponse)
async def enhanced_chat(request: ChatRequest):
    """Enhanced chat endpoint with conversation context and system instructions"""
    try:
        if not enhanced_rag_system:
            raise HTTPException(status_code=500, detail="Enhanced RAG system not initialized")
        
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        logger.info(f"Processing enhanced chat request: {request.message} (session: {request.session_id})")
        
        # Use enhanced system with conversation context
        result = await enhanced_rag_system.chat_with_context(
            request.message, 
            request.session_id or "default-session"
        )
        
        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            session_id=result.get("session_id"),
            conversation_length=result.get("conversation_length")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing enhanced chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process enhanced chat request: {str(e)}")

@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a conversation session"""
    try:
        if enhanced_rag_system:
            info = enhanced_rag_system.get_session_info(session_id)
            return info
        else:
            return {"error": "Enhanced RAG system not available"}
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session info: {str(e)}")
        
        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")

@app.get("/api/docs-count")
async def get_docs_count():
    """Get the number of indexed documents"""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        # Query the database for document count
        result = rag_system.supabase.table("documents").select("id", count="exact").execute()
        
        return {
            "total_chunks": result.count,
            "message": "Document count retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting document count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document count: {str(e)}")

@app.delete("/api/clear")
async def clear_documents():
    """Clear all documents from the database"""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        # Clear all documents
        result = rag_system.supabase.table("documents").delete().neq("id", 0).execute()
        
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG Chat API...")
    print("📚 Visit http://localhost:8000/docs for interactive API documentation")
    print("🔍 Test endpoints:")
    print("   GET  /api/health - Check system status")
    print("   POST /api/index - Index a GitHub repository")
    print("   POST /api/chat - Ask questions about indexed content")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
