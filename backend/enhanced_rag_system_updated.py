import os
import logging
from typing import List, Dict, Any, Optional
import tempfile
import shutil
import json
from pathlib import Path
import uuid
from datetime import datetime
import asyncio

# Core imports
from dotenv import load_dotenv

# Supabase
from supabase import create_client, Client

# LangChain imports
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain imports not available: {e}")
    LANGCHAIN_AVAILABLE = False

# Google AI SDK
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError as e:
    print(f"Google GenerativeAI not available: {e}")
    GENAI_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)

class ConversationManager:
    """Enhanced conversation manager with better context memory"""
    
    def __init__(self, supabase_client: Optional[Client] = None):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.max_context_messages = 20  # Increased from 10 to 20
        self.max_context_length = 4000  # Maximum context length in characters
        self.supabase = supabase_client
        
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new conversation session with better validation"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Check if session already exists
        if session_id in self.sessions:
            logger.info(f"Session {session_id} already exists, returning existing session")
            return session_id
            
        self.sessions[session_id] = {
            'messages': [],
            'created_at': datetime.now(),
            'last_updated': datetime.now(),
            'repository_context': None,
            'last_repository': None,
            'message_count': 0
        }
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message with improved error handling and context management"""
        try:
            # Ensure session exists
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found, creating new session")
                session_id = self.create_session(session_id)
            
            message = {
                'role': role,  # 'user' or 'assistant'
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {},
                'message_id': str(uuid.uuid4())
            }
            
            self.sessions[session_id]['messages'].append(message)
            self.sessions[session_id]['last_updated'] = datetime.now()
            self.sessions[session_id]['message_count'] += 1
            
            # Smart context management - keep important messages
            self._manage_context(session_id)
            
            logger.debug(f"Added {role} message to session {session_id}")
            
        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {e}")
            # Try to recover by creating new session
            try:
                self.create_session(session_id)
                self.add_message(session_id, role, content, metadata)
            except:
                logger.error(f"Failed to recover session {session_id}")
    
    def _manage_context(self, session_id: str):
        """Intelligent context management to keep relevant messages"""
        if session_id not in self.sessions:
            return
            
        messages = self.sessions[session_id]['messages']
        
        # If we have too many messages, keep the most recent and important ones
        if len(messages) > self.max_context_messages:
            # Always keep the first message (often contains important context)
            first_message = messages[0] if messages else None
            
            # Keep the most recent messages
            recent_messages = messages[-(self.max_context_messages - 1):]
            
            # Combine first message with recent messages
            if first_message and first_message not in recent_messages:
                self.sessions[session_id]['messages'] = [first_message] + recent_messages
            else:
                self.sessions[session_id]['messages'] = recent_messages
    
    def get_conversation_context(self, session_id: str) -> str:
        """Get well-formatted conversation history for context"""
        if session_id not in self.sessions:
            return ""
        
        messages = self.sessions[session_id]['messages']
        if not messages:
            return ""
        
        # Build context with proper formatting
        context_parts = ["=== CONVERSATION HISTORY ==="]
        total_length = 0
        
        # Work backwards from most recent messages
        for msg in reversed(messages[-10:]):  # Last 10 messages max
            role = "User" if msg['role'] == 'user' else "Assistant"
            content = msg['content']
            
            # Truncate very long messages but keep them meaningful
            if len(content) > 300:
                content = content[:300] + "... [truncated]"
            
            message_text = f"{role}: {content}"
            
            # Check if adding this message would exceed length limit
            if total_length + len(message_text) > self.max_context_length:
                break
                
            context_parts.insert(1, message_text)  # Insert after header
            total_length += len(message_text)
        
        context_parts.append("=== END CONVERSATION HISTORY ===")
        return "\n\n".join(context_parts)
    
    def update_repository_context(self, session_id: str, repo_url: str, repo_info: Dict):
        """Update repository context with better validation"""
        if session_id not in self.sessions:
            session_id = self.create_session(session_id)
        
        self.sessions[session_id]['repository_context'] = repo_info
        self.sessions[session_id]['last_repository'] = repo_url
        self.sessions[session_id]['last_updated'] = datetime.now()
        
        logger.info(f"Updated repository context for session {session_id}: {repo_url}")
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information"""
        return self.sessions.get(session_id)
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
            return True
        return False

class EnhancedRAGSystem:
    """Enhanced RAG system with improved conversation memory and context management"""
    
    def __init__(self):
        # Environment variables
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY") 
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not all([self.supabase_url, self.supabase_key, self.gemini_api_key]):
            raise ValueError("Missing required environment variables: SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY")
        
        # Initialize Supabase client
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise e
        
        # Initialize conversation manager with Supabase client
        self.conversation_manager = ConversationManager(self.supabase)
        
        # Initialize Google AI client
        if GENAI_AVAILABLE:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
                logger.info("Google AI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Google AI client: {e}")
                raise e
        
        # Initialize LangChain components if available
        if LANGCHAIN_AVAILABLE:
            try:
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=self.gemini_api_key
                )
                
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                logger.info("LangChain components initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LangChain components: {e}")
                raise e
        else:
            logger.warning("LangChain not available - limited functionality")
        
        # Set table name
        self.table_name = "documents"

    def _get_system_instructions(self) -> str:
        """Improved system instructions with better context awareness"""
        return """You are an expert AI Code Repository Assistant with perfect conversation memory and deep analytical capabilities.

CORE PRINCIPLES:
1. PERFECT MEMORY: Never contradict yourself within a conversation. Remember everything discussed.
2. CONTEXT AWARENESS: Use conversation history to provide consistent, building responses.
3. INTELLIGENT ANALYSIS: Understand code patterns, project structure, and developer intent.
4. HELPFUL GUIDANCE: Provide practical, actionable advice and explanations.

CONVERSATION MEMORY:
- You have access to our entire conversation history
- Reference previous discussions when relevant
- Build upon earlier explanations
- Maintain consistency in your responses
- Remember user preferences and context

ANALYSIS APPROACH:
- Understand the project type and architecture
- Recognize technology stacks and patterns
- Provide contextual explanations
- Connect related concepts across the codebase
- Offer practical insights and suggestions

RESPONSE STYLE:
- Be conversational but professional
- Provide clear, actionable information
- Use examples when helpful
- Structure responses logically
- Remember what you've already explained

Always use the conversation history to maintain context and consistency."""

    async def chat_with_context(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced chat function with improved context memory"""
        try:
            # Create or validate session
            if not session_id:
                session_id = self.conversation_manager.create_session()
            elif session_id not in self.conversation_manager.sessions:
                logger.info(f"Session {session_id} not found, creating new session")
                session_id = self.conversation_manager.create_session(session_id)
            
            # Add user message to conversation
            self.conversation_manager.add_message(session_id, 'user', message)
            
            # Get conversation context
            conversation_context = self.conversation_manager.get_conversation_context(session_id)
            
            # Get relevant documents using vector search
            relevant_docs = []
            search_results = {}
            
            if LANGCHAIN_AVAILABLE:
                try:
                    question_embedding = await asyncio.to_thread(
                        self.embeddings.embed_query, message
                    )
                    search_results = await self._enhanced_vector_search(question_embedding, message, session_id)
                    relevant_docs = search_results.get('documents', [])
                except Exception as e:
                    logger.error(f"Vector search failed: {e}")
                    # Fallback to text search
                    search_results = await self._fallback_text_search(message)
                    relevant_docs = search_results.get('documents', [])
            
            # Prepare enhanced context
            context_info = self._prepare_enhanced_context(
                relevant_docs, 
                message, 
                conversation_context, 
                session_id
            )
            
            # Generate response using Google AI
            if GENAI_AVAILABLE:
                full_prompt = f"""{self._get_system_instructions()}

{context_info}

CURRENT QUESTION: {message}

Please respond using the conversation history to maintain consistency and the repository context to provide accurate information."""
                
                try:
                    response = await asyncio.to_thread(
                        self.model.generate_content,
                        full_prompt
                    )
                    answer = response.text
                except Exception as e:
                    logger.error(f"Gemini API error: {e}")
                    answer = f"I encountered an error generating a response: {str(e)}"
            else:
                answer = "Google AI client not available. Please check your API configuration."
            
            # Add assistant response to conversation
            self.conversation_manager.add_message(
                session_id, 
                'assistant', 
                answer,
                {
                    'sources_count': len(relevant_docs),
                    'search_method': search_results.get('method', 'none'),
                    'total_results': search_results.get('total', 0)
                }
            )
            
            # Prepare sources for response
            sources = []
            for doc in relevant_docs[:3]:  # Top 3 sources
                source_preview = doc.get("content", "")[:200]
                if len(source_preview) < len(doc.get("content", "")):
                    source_preview += "..."
                sources.append({
                    'content': source_preview,
                    'file_path': doc.get('file_path', 'Unknown'),
                    'repository': doc.get('repository', 'Unknown')
                })
            
            # Get session info
            session_info = self.conversation_manager.get_session_info(session_id)
            conversation_length = session_info.get('message_count', 0) if session_info else 0
            
            return {
                "answer": answer,
                "sources": sources,
                "session_id": session_id,
                "conversation_length": conversation_length,
                "search_info": {
                    "method": search_results.get('method', 'none'),
                    "results_count": len(relevant_docs)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in chat_with_context: {e}", exc_info=True)
            
            # Ensure we return a valid session_id even in error cases
            if not session_id:
                session_id = self.conversation_manager.create_session()
            
            # Log the error to conversation history
            try:
                self.conversation_manager.add_message(
                    session_id, 
                    'system', 
                    f"Error occurred: {str(e)}",
                    {'error': True}
                )
            except:
                pass  # Don't fail if we can't log the error
            
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}. Please try again or rephrase your question.",
                "sources": [],
                "session_id": session_id,
                "conversation_length": 0,
                "error": True
            }

    async def _enhanced_vector_search(self, query_embedding: List[float], query: str, session_id: str) -> Dict[str, Any]:
        """Enhanced vector search with better error handling"""
        try:
            # Perform vector search using Supabase
            response = self.supabase.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.3,  # Lowered threshold for more results
                    'match_count': 10
                }
            ).execute()
            
            documents = response.data if response.data else []
            
            logger.info(f"Vector search found {len(documents)} documents for session {session_id}")
            
            return {
                'documents': documents,
                'method': 'vector',
                'total': len(documents)
            }
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return {
                'documents': [],
                'method': 'vector_failed',
                'total': 0,
                'error': str(e)
            }

    async def _fallback_text_search(self, query: str) -> Dict[str, Any]:
        """Fallback text search when vector search fails"""
        try:
            # Simple text search in the content field
            response = self.supabase.table(self.table_name).select("*").ilike(
                'content', f'%{query}%'
            ).limit(5).execute()
            
            documents = response.data if response.data else []
            
            logger.info(f"Text search found {len(documents)} documents")
            
            return {
                'documents': documents,
                'method': 'text',
                'total': len(documents)
            }
            
        except Exception as e:
            logger.error(f"Text search error: {e}")
            return {
                'documents': [],
                'method': 'text_failed',
                'total': 0,
                'error': str(e)
            }

    def _prepare_enhanced_context(self, documents: List[Dict], query: str, conversation_context: str, session_id: str) -> str:
        """Prepare enhanced context with better formatting"""
        context_parts = []
        
        # Add conversation context
        if conversation_context:
            context_parts.append(conversation_context)
        
        # Add repository context
        if documents:
            context_parts.append("\n=== RELEVANT REPOSITORY CONTENT ===")
            
            for i, doc in enumerate(documents[:5], 1):  # Top 5 documents
                file_path = doc.get('file_path', 'Unknown file')
                content = doc.get('content', '')
                
                # Truncate very long content
                if len(content) > 1000:
                    content = content[:1000] + "... [content truncated]"
                
                context_parts.append(f"\n--- Source {i}: {file_path} ---")
                context_parts.append(content)
            
            context_parts.append("\n=== END REPOSITORY CONTENT ===\n")
        else:
            context_parts.append("\n=== NO SPECIFIC REPOSITORY CONTENT FOUND ===")
            context_parts.append("Provide general guidance based on the conversation context.\n")
        
        return "\n".join(context_parts)

    # Original methods preserved for compatibility
    async def index_github_repo(self, repo_url: str, max_files: int = 100) -> Dict[str, Any]:
        """Index a GitHub repository - keeping original functionality"""
        from rag_system import RAGSystem
        original_rag = RAGSystem()
        return await original_rag.index_github_repo(repo_url, max_files)

    async def _get_recent_repository_docs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get documents from the most recently indexed repository"""
        try:
            recent_source = self.supabase.table(self.table_name)\
                .select('source_url')\
                .order('id', desc=True)\
                .limit(1)\
                .execute()
            
            if not recent_source.data:
                return []
            
            latest_source_url = recent_source.data[0]['source_url']
            
            result = self.supabase.table(self.table_name)\
                .select('content, metadata, source_url')\
                .eq('source_url', latest_source_url)\
                .limit(limit)\
                .execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Failed to get recent repository docs: {e}")
            return []

    async def get_document_count(self) -> int:
        """Get total number of documents"""
        try:
            result = self.supabase.table(self.table_name).select('id', count='exact').execute()
            return result.count or 0
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0

    def get_session_info_detailed(self, session_id: str) -> Dict[str, Any]:
        """Get detailed information about a conversation session"""
        if session_id not in self.conversation_manager.sessions:
            return {"error": "Session not found"}
        
        session = self.conversation_manager.sessions[session_id]
        return {
            "session_id": session_id,
            "message_count": session.get('message_count', len(session['messages'])),
            "created_at": session['created_at'].isoformat(),
            "last_updated": session.get('last_updated', session['created_at']).isoformat(),
            "last_repository": session.get('last_repository'),
            "repository_context": session.get('repository_context')
        }

if __name__ == "__main__":
    # Test the enhanced RAG system
    import asyncio
    
    async def test_enhanced_rag():
        try:
            rag = EnhancedRAGSystem()
            print("✅ Enhanced RAG System initialized successfully")
            
            # Test conversation memory
            session_id = rag.conversation_manager.create_session()
            print(f"✅ Created test session: {session_id}")
            
            # Test adding messages
            rag.conversation_manager.add_message(session_id, 'user', 'Hello, can you help me understand this repository?')
            rag.conversation_manager.add_message(session_id, 'assistant', 'Of course! I can help you understand the repository. What specific aspects would you like to explore?')
            
            # Test getting context
            context = rag.conversation_manager.get_conversation_context(session_id)
            print(f"✅ Context retrieved: {len(context)} characters")
            
            print("✅ All enhanced context memory tests passed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    asyncio.run(test_enhanced_rag())
