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
    """Manages conversation context and memory"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}        self.max_context_messages = 10  # Keep last 10 exchanges
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new conversation session"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            'messages': [],
            'created_at': datetime.now(),
            'repository_context': None,
            'last_repository': None
        }
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation history"""
        try:
            if session_id not in self.sessions:
                session_id = self.create_session(session_id)
            
            message = {
                'role': role,  # 'user' or 'assistant'
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            self.sessions[session_id]['messages'].append(message)
            
            # Keep only recent messages to avoid token limits
            if len(self.sessions[session_id]['messages']) > self.max_context_messages * 2:
                self.sessions[session_id]['messages'] = self.sessions[session_id]['messages'][-self.max_context_messages * 2:]
        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {e}")
            # Create a new session if there's an error
            self.create_session(session_id)
    
    def get_conversation_context(self, session_id: str) -> str:
        """Get formatted conversation history for context"""
        if session_id not in self.sessions:
            return ""
        
        messages = self.sessions[session_id]['messages']
        if not messages:
            return ""
        
        context = "CONVERSATION HISTORY:\n"
        for msg in messages[-6:]:  # Last 6 messages (3 exchanges)
            role = "Human" if msg['role'] == 'user' else "Assistant"
            context += f"{role}: {msg['content'][:200]}...\n"
        
        return context
    
    def update_repository_context(self, session_id: str, repo_url: str, repo_info: Dict):
        """Update the current repository context for the session"""
        if session_id not in self.sessions:
            session_id = self.create_session()
        
        self.sessions[session_id]['repository_context'] = repo_info
        self.sessions[session_id]['last_repository'] = repo_url

class EnhancedRAGSystem:
    """Enhanced RAG system with conversation memory and detailed instructions"""
    
    def __init__(self):
        # Environment variables
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY") 
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not all([self.supabase_url, self.supabase_key, self.gemini_api_key]):
            raise ValueError("Missing required environment variables: SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY")
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager()
        
        # Initialize Supabase client
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise e
          # Initialize Google AI client
        if GENAI_AVAILABLE:
            try:
                genai.configure(api_key=self.gemini_api_key)
                # Create model without system_instruction (will add it to each prompt instead)
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
        """Comprehensive system instructions for the AI assistant"""
        return """
# ROLE & IDENTITY
You are an expert AI Code Repository Assistant with deep analytical capabilities and perfect memory within conversations. You specialize in understanding, analyzing, and explaining software projects, codebases, and development concepts.

# CORE PRINCIPLES

## 1. CONSISTENCY & MEMORY
- NEVER contradict yourself within the same conversation
- Remember what you've said before and build upon it
- If you previously stated something is present, don't later say it's missing
- Reference previous parts of our conversation when relevant
- Maintain logical consistency throughout the entire session

## 2. INTELLIGENT ANALYSIS
- Don't just search - UNDERSTAND and SYNTHESIZE information
- Connect dots between different pieces of information
- Use deductive reasoning to infer missing information
- Analyze patterns in code structure, naming conventions, and project organization

## 3. CONTEXTUAL AWARENESS
- Understand the type of project (web app, API, library, CLI tool, etc.)
- Recognize technology stacks and frameworks
- Infer project purpose from file structure and dependencies
- Consider the user's perspective and intent

# ANALYTICAL FRAMEWORK

## When analyzing repositories:

### 1. PROJECT IDENTIFICATION
- Determine project type from package.json, requirements.txt, Cargo.toml, etc.
- Identify main technologies: React, Next.js, FastAPI, Django, etc.
- Recognize architectural patterns: MVC, microservices, monolith, etc.

### 2. FILE STRUCTURE ANALYSIS
- README files → Project documentation and purpose
- Config files → Technology stack and deployment setup
- Source directories → Core functionality and organization
- Test directories → Testing strategy and coverage
- Build files → Compilation and deployment process

### 3. CODE UNDERSTANDING
- Analyze function/class purposes and relationships
- Identify design patterns and best practices
- Understand data flow and architecture
- Recognize performance optimizations and security measures

### 4. DEPENDENCY ANALYSIS
- Understand external libraries and their purposes
- Identify potential security vulnerabilities
- Suggest improvements or alternatives

# RESPONSE GUIDELINES

## Question Types & Approaches:

### CODE QUESTIONS
- Explain code functionality in plain language
- Identify the purpose and context
- Highlight important patterns or techniques
- Suggest improvements if asked

### SETUP/INSTALLATION QUESTIONS  
- Provide step-by-step instructions
- Mention prerequisites and dependencies
- Include troubleshooting tips
- Consider different environments (Windows, Mac, Linux)

### ARCHITECTURE QUESTIONS
- Explain system design and component relationships
- Identify data flow and communication patterns
- Discuss scalability and performance considerations
- Suggest architectural improvements

### DEBUGGING QUESTIONS
- Analyze error messages and symptoms
- Suggest systematic debugging approaches
- Recommend tools and techniques
- Provide preventive measures

## Response Structure:

### 1. DIRECT ANSWER
Start with a clear, direct response to the question

### 2. SUPPORTING EVIDENCE
Reference specific files, code snippets, or configurations that support your answer

### 3. CONTEXTUAL INSIGHTS
Explain why things are structured this way and the benefits/trade-offs

### 4. ACTIONABLE RECOMMENDATIONS
Provide next steps, improvements, or related suggestions

### 5. CONFIDENCE LEVEL
If uncertain, clearly state your confidence level and what additional information would help

# CONVERSATION MEMORY USAGE

## Reference Previous Context:
- "As I mentioned earlier about..."
- "Building on our previous discussion..."
- "Consistent with what we found in [filename]..."
- "This relates to the [technology] setup we discussed..."

## Maintain Consistency:
- Keep track of facts established in conversation
- Build upon previous analyses
- Don't repeat lengthy explanations already given
- Reference earlier findings to support new insights

# TECHNICAL EXPERTISE AREAS

## Languages & Frameworks:
- JavaScript/TypeScript (React, Next.js, Node.js, Express)
- Python (Django, FastAPI, Flask, Data Science)
- Web Technologies (HTML, CSS, Tailwind, Bootstrap)
- Databases (SQL, NoSQL, Vector databases)
- Cloud & DevOps (AWS, Azure, Docker, Kubernetes)

## Development Practices:
- Version Control (Git workflows)
- Testing (Unit, Integration, E2E)
- CI/CD pipelines
- Code quality and security
- Performance optimization

# SPECIAL CAPABILITIES

## Repository Analysis:
- Determine project purpose from minimal information
- Identify missing documentation or setup steps
- Suggest project improvements and best practices
- Explain complex architectural decisions

## Code Review:
- Identify bugs and security issues
- Suggest performance improvements
- Recommend better practices
- Explain complex algorithms and patterns

## Learning Support:
- Break down complex concepts
- Provide learning paths and resources
- Explain "why" not just "how"
- Adapt explanations to user's experience level

# IMPORTANT REMINDERS

1. **ALWAYS be consistent** - If you said a file exists, don't later say it doesn't
2. **USE MEMORY** - Reference previous parts of our conversation
3. **BE SPECIFIC** - Use exact file names, line numbers, and code snippets when possible
4. **EXPLAIN REASONING** - Show your thought process and how you reached conclusions
5. **ACKNOWLEDGE UNCERTAINTY** - Say when you're not sure and explain what would help clarify

Remember: You're not just answering questions - you're having an intelligent conversation about code and helping the user understand their project deeply.
"""

    async def chat_with_context(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced chat function with conversation context"""
        try:
            # Create session if not provided
            if not session_id:
                session_id = self.conversation_manager.create_session()
            
            # Add user message to conversation
            self.conversation_manager.add_message(session_id, 'user', message)
            
            # Get conversation context
            conversation_context = self.conversation_manager.get_conversation_context(session_id)
            
            # Get relevant documents using vector search
            relevant_docs = []
            if LANGCHAIN_AVAILABLE:
                question_embedding = self.embeddings.embed_query(message)
                relevant_docs = await self._enhanced_vector_search(question_embedding, message, session_id)
            
            # Prepare enhanced context
            enhanced_context = self._prepare_enhanced_context(
                relevant_docs, 
                message, 
                conversation_context, 
                session_id
            )
              # Generate response using Google AI
            if GENAI_AVAILABLE:
                full_prompt = f"""
{self._get_system_instructions()}

{enhanced_context}

CURRENT QUESTION: {message}

Respond as the expert AI assistant following the system instructions above. Use the conversation history to maintain consistency and the repository context to provide accurate information.
"""
                
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    full_prompt
                )
                
                answer = response.text
            else:
                answer = "Google AI client not available"
            
            # Add assistant response to conversation
            self.conversation_manager.add_message(
                session_id, 
                'assistant', 
                answer,
                {'sources_count': len(relevant_docs)}
            )
              # Prepare sources for response
            sources = [doc.get("content", "")[:200] + "..." for doc in relevant_docs[:3]]
            
            # Get conversation length safely
            conversation_length = 0
            if session_id in self.conversation_manager.sessions:
                conversation_length = len(self.conversation_manager.sessions[session_id]['messages'])
            
            return {
                "answer": answer,
                "sources": sources,
                "session_id": session_id,
                "conversation_length": conversation_length
            }
            
        except Exception as e:
            logger.error(f"Error in chat_with_context: {e}")
            # Ensure session_id is available even in error case
            if not session_id:
                session_id = "error-session"
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "session_id": session_id,
                "error": str(e)
            }

    async def _enhanced_vector_search(self, query_embedding: List[float], question: str, session_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Enhanced vector search with context awareness"""
        try:
            # Get vector search results
            result = self.supabase.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.75,
                    'match_count': limit * 2  # Get more initially to filter
                }
            ).execute()
            
            docs = result.data or []
            
            # If asking about "this repo" or similar, prioritize recent repository
            if any(phrase in question.lower() for phrase in 
                   ["this repo", "this repository", "this project", "current repo"]):
                recent_docs = await self._get_recent_repository_docs(limit=6)
                if recent_docs:
                    # Update session context
                    if recent_docs:
                        repo_url = recent_docs[0].get('source_url', 'Unknown')
                        self.conversation_manager.update_repository_context(
                            session_id, 
                            repo_url, 
                            {'doc_count': len(recent_docs)}
                        )
                    docs = recent_docs + docs
              # Enhanced filtering and ranking
            filtered_docs = self._filter_and_rank_docs(docs, question)
            
            return filtered_docs[:limit]
            
        except Exception as e:
            logger.error(f"Enhanced vector search failed: {e}")
            return []

    def _filter_and_rank_docs(self, docs: List[Dict], question: str) -> List[Dict]:
        """Filter and rank documents based on relevance and question type"""
        
        if not docs:
            return []
        
        question_lower = question.lower()
        
        # Define relevance boosters
        file_type_boosters = {
            'readme': 3.0,
            'documentation': 2.5,
            'config': 2.0,
            'main': 2.0,
            'index': 1.5,
            '.md': 2.0,
            '.json': 1.5,
            '.py': 1.3,
            '.js': 1.3,
            '.ts': 1.3
        }
        
        question_type_boosters = {
            'setup': ['readme', 'install', 'requirements', 'package.json'],
            'config': ['config', '.env', 'settings', '.json', '.yaml'],
            'code': ['.py', '.js', '.ts', '.java', '.cpp'],
            'purpose': ['readme', 'about', 'description', 'main'],
            'error': ['debug', 'error', 'log', 'traceback']
        }
        
        # Score each document
        scored_docs = []
        for doc in docs:
            try:
                score = 1.0
                
                # Handle different document formats
                if isinstance(doc, dict):
                    # Get source from metadata or directly
                    metadata = doc.get('metadata', {})
                    if isinstance(metadata, str):
                        # If metadata is a JSON string, try to parse it
                        try:
                            import json
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                    
                    source = ''
                    if isinstance(metadata, dict):
                        source = metadata.get('source', '')
                    
                    # Fallback: check if source_url exists
                    if not source:
                        source = doc.get('source_url', '')
                    
                    content = doc.get('content', '').lower()
                else:
                    # Handle unexpected format
                    logger.warning(f"Unexpected document format: {type(doc)}")
                    continue
                
                # File type scoring
                for file_type, boost in file_type_boosters.items():
                    if file_type.lower() in source.lower():
                        score *= boost
                        break
                
                # Question type scoring
                for q_type, keywords in question_type_boosters.items():
                    if any(keyword in question_lower for keyword in [q_type]):
                        for keyword in keywords:
                            if keyword.lower() in source.lower():
                                score *= 1.8
                                break
                
                # Content relevance (basic keyword matching)
                if content:
                    question_words = set(question_lower.split())
                    content_words = set(content.split())
                    overlap = len(question_words.intersection(content_words))
                    if overlap > 0:
                        score *= (1 + overlap * 0.1)
                
                scored_docs.append((score, doc))
                
            except Exception as e:
                logger.warning(f"Error processing document in ranking: {e}")
                # Include the document with default score
                scored_docs.append((1.0, doc))
        
        # Sort by score and return
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs]

    def _prepare_enhanced_context(self, docs: List[Dict], question: str, conversation_context: str, session_id: str) -> str:
        """Prepare comprehensive context for the AI"""
        
        # Analyze available documents
        doc_analysis = self._analyze_documents(docs)
        
        # Get session info
        session_info = ""
        if session_id in self.conversation_manager.sessions:
            session = self.conversation_manager.sessions[session_id]
            if session.get('last_repository'):
                session_info = f"CURRENT REPOSITORY: {session['last_repository']}\n"
        
        context = f"""
{session_info}

{conversation_context}

REPOSITORY ANALYSIS:
{doc_analysis}

RELEVANT DOCUMENTS:
"""
        
        for i, doc in enumerate(docs[:6], 1):
            source = doc.get('metadata', {}).get('source', 'Unknown file')
            content = doc.get('content', '')[:800]  # Limit content length
            context += f"\n--- Document {i}: {source} ---\n{content}\n"
        
        context += f"""

ANALYSIS GUIDELINES:
- Use the conversation history to maintain consistency
- Reference previous discussions when relevant  
- If you've made statements before, build upon them rather than contradicting
- Focus on the current repository context when answering about "this repo/project"
- Provide specific, actionable insights based on the available code and configuration
"""
        
        return context

    def _analyze_documents(self, docs: List[Dict]) -> str:
        """Analyze the available documents to provide context"""
        if not docs:
            return "No relevant documents found in the current context."
        
        # Count file types
        file_types = {}
        repositories = set()
        
        for doc in docs:
            metadata = doc.get('metadata', {})
            source = metadata.get('source', '')
            source_url = doc.get('source_url', '')
            
            if source_url:
                repositories.add(source_url)
            
            if '.' in source:
                ext = source.split('.')[-1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1
        
        analysis = f"""
- Documents available: {len(docs)}
- Repositories: {len(repositories)} ({', '.join(list(repositories)[:2])})
- File types: {', '.join(f'{ext}({count})' for ext, count in file_types.items())}
- Primary content: {max(file_types.items(), key=lambda x: x[1])[0] if file_types else 'mixed'}
"""
        
        return analysis

    # Include the original methods for compatibility
    async def index_github_repo(self, repo_url: str, max_files: int = 100) -> Dict[str, Any]:
        """Index a GitHub repository - keeping original functionality"""
        # This would be the same as your original implementation
        # Delegating to avoid duplication
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

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a conversation session"""
        if session_id not in self.conversation_manager.sessions:
            return {"error": "Session not found"}
        
        session = self.conversation_manager.sessions[session_id]
        return {
            "session_id": session_id,
            "message_count": len(session['messages']),
            "created_at": session['created_at'].isoformat(),
            "last_repository": session.get('last_repository'),
            "repository_context": session.get('repository_context')
        }
