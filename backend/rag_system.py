import os
import logging
from typing import List, Dict, Any
import tempfile
import shutil
import json
from pathlib import Path

# Core imports
from dotenv import load_dotenv
# Remove git import from here - will import lazily when needed

# Supabase
from supabase import create_client, Client

# LangChain imports (we'll use these step by step)
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain imports not available: {e}")
    LANGCHAIN_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)

class RAGSystem:
    """Production RAG system for GitHub repository indexing and chat"""
    
    def __init__(self):
        # Environment variables
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY") 
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not all([self.supabase_url, self.supabase_key, self.gemini_api_key]):
            raise ValueError("Missing required environment variables: SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY")
        
        # Initialize Supabase client - simplified approach
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise e
        
        # Initialize LangChain components if available
        if LANGCHAIN_AVAILABLE:
            try:
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=self.gemini_api_key
                )
                
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-exp",
                    google_api_key=self.gemini_api_key,
                    temperature=0.1
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

    async def index_github_repo(self, repo_url: str, max_files: int = 100) -> Dict[str, Any]:
        """Index a GitHub repository by downloading and processing its contents"""
        try:
            if not LANGCHAIN_AVAILABLE:
                return {"success": False, "error": "LangChain not available"}
            
            logger.info(f"Starting to index repository: {repo_url}")
            
            # Parse GitHub URL to get owner/repo
            if "github.com" not in repo_url:
                raise ValueError("Invalid GitHub URL")
            
            # Extract owner/repo from URL
            parts = repo_url.replace("https://github.com/", "").replace("http://github.com/", "").split("/")
            if len(parts) < 2:
                raise ValueError("Invalid GitHub repository URL format")
            
            owner, repo = parts[0], parts[1]
            
            # Download and process the repository
            documents = await self._download_and_process_repo(owner, repo, max_files)
            
            if not documents:
                return {"success": False, "error": "No documents found in repository"}
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            
            # Store in Supabase
            await self._store_chunks(chunks, repo_url)
            
            return {
                "success": True, 
                "documents_processed": len(documents),
                "chunks_created": len(chunks),
                "repo_url": repo_url
            }
            
        except Exception as e:
            logger.error(f"Failed to index repository {repo_url}: {e}")
            return {"success": False, "error": str(e)}

    async def _download_and_process_repo(self, owner: str, repo: str, max_files: int) -> List[Document]:
        """Download and process a GitHub repository"""
        documents = []
        
        try:
            # Dynamic import to avoid Windows multiprocessing issues
            from git import Repo
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_path = Path(temp_dir) / repo
                
                # Clone the repository
                logger.info(f"Cloning repository {owner}/{repo}")
                git_url = f"https://github.com/{owner}/{repo}.git"
                Repo.clone_from(git_url, repo_path, depth=1)
                
                # Process files
                file_count = 0
                for file_path in repo_path.rglob("*"):
                    if file_count >= max_files:
                        break
                        
                    if file_path.is_file():
                        try:
                            # Skip binary files and common non-text files
                            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.tar', '.gz']:
                                continue
                            
                            # Read file content
                            try:
                                content = file_path.read_text(encoding='utf-8')
                            except UnicodeDecodeError:
                                # Try with latin-1 encoding as fallback
                                content = file_path.read_text(encoding='latin-1')
                            
                            # Create document with metadata
                            relative_path = file_path.relative_to(repo_path)
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source": str(relative_path),
                                    "repository": f"{owner}/{repo}",
                                    "file_type": file_path.suffix,
                                    "size": len(content)
                                }
                            )
                            documents.append(doc)
                            file_count += 1
                            
                            if file_count % 10 == 0:
                                logger.info(f"Processed {file_count} files")
                    
                        except Exception as e:
                            logger.warning(f"Failed to load {file_path}: {e}")
                            continue
        
        except Exception as e:
            logger.error(f"Failed to download repository: {e}")
            raise e
        
        return documents

    async def _store_chunks(self, chunks: List[Document], source_url: str):
        """Store document chunks in Supabase"""
        try:
            # Clear existing data for this source
            logger.info(f"Clearing existing data for {source_url}")
            self.supabase.table(self.table_name).delete().eq('source_url', source_url).execute()
            
            # Prepare records for insertion
            records = []
            
            for i, chunk in enumerate(chunks):
                # Generate embedding for this chunk if LangChain is available
                embedding = None
                if LANGCHAIN_AVAILABLE:
                    try:
                        logger.info(f"Generating embedding for chunk {i}/{len(chunks)} (content length: {len(chunk.page_content)})")
                        embedding = self.embeddings.embed_query(chunk.page_content)
                        logger.info(f"✅ Generated embedding for chunk {i}, length: {len(embedding) if embedding else 'None'}")
                    except Exception as e:
                        logger.warning(f"❌ Failed to generate embedding for chunk {i}: {e}")
                else:
                    logger.warning(f"⚠️ LangChain not available, skipping embedding for chunk {i}")
                
                record = {
                    'content': chunk.page_content[:4000],  # Limit content size
                    'metadata': json.dumps(chunk.metadata),
                    'source_url': source_url,
                    'chunk_index': i
                }
                
                # Add embedding if available
                if embedding:
                    record['embedding'] = embedding
                    logger.info(f"✅ Added embedding to record for chunk {i}")
                else:
                    logger.warning(f"⚠️ No embedding available for chunk {i}")
                    
                records.append(record)
            
            # Insert in small batches to avoid timeouts
            batch_size = 5
            successful_inserts = 0
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                try:
                    result = self.supabase.table(self.table_name).insert(batch).execute()
                    successful_inserts += len(batch)
                    logger.info(f"Stored batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size}")
                except Exception as e:
                    logger.warning(f"Failed to store batch {i//batch_size + 1}: {e}")
            
            logger.info(f"Successfully stored {successful_inserts}/{len(records)} chunks")
            
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            raise e

    async def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using retrieved documents with vector similarity search"""
        try:
            if not LANGCHAIN_AVAILABLE:
                return {"answer": "LangChain not available", "sources": []}
            
            # Generate embedding for the user's question
            question_embedding = self.embeddings.embed_query(question)
            logger.info(f"Generated question embedding, length: {len(question_embedding)}")
              # Search for similar documents using vector similarity
            similar_docs = await self._vector_search(question_embedding, limit=5)
            logger.info(f"Vector search returned {len(similar_docs)} documents")
            
            # If asking about "this repo" or repository purpose, also get recent repository content
            if any(phrase in question.lower() for phrase in ["this repo", "this repository", "purpose of this", "what is this"]):
                logger.info("Question seems to be about a specific repository, adding recent content")
                recent_docs = await self._get_recent_repository_docs(limit=5)
                if recent_docs:
                    logger.info(f"Found {len(recent_docs)} documents from most recent repository")
                    # Prioritize recent docs by putting them first
                    similar_docs = recent_docs + similar_docs
            
            # If vector search doesn't return results, try text search as fallback
            if not similar_docs:
                logger.info("Vector search returned no results, trying text search fallback")
                similar_docs = await self._text_search(question, limit=5)
                logger.info(f"Text search returned {len(similar_docs)} documents")
            
            if not similar_docs:
                return {"answer": "I couldn't find relevant information to answer your question.", "sources": []}
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc["content"] for doc in similar_docs])
            logger.info(f"Prepared context with {len(context)} characters from {len(similar_docs)} documents")
              # Create prompt template
            prompt_template = ChatPromptTemplate.from_template("""
            Based on the provided context from GitHub repository content, answer the following question. 
            
            If the question is about "this repository" or "this repo", focus on the most recent repository content provided.
            Look for README files, documentation, code comments, configuration files, and project structure to understand the repository's purpose.
            
            If the context doesn't contain enough information to answer the question, say so clearly.

            Context:
            {context}

            Question: {question}

            Answer:
            """)
            
            # Generate answer using LLM
            chain = prompt_template | self.llm | StrOutputParser()
            answer = await chain.ainvoke({"context": context, "question": question})
            
            # Return answer with sources
            sources = [doc["content"][:200] + "..." for doc in similar_docs]
            
            return {"answer": answer, "sources": sources}
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {"answer": f"Error processing question: {str(e)}", "sources": []}

    async def _vector_search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        try:
            result = self.supabase.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.78,
                    'match_count': limit
                }
            ).execute()
            
            logger.info(f"Vector search query executed, returned {len(result.data) if result.data else 0} results")
            return result.data or []
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _text_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fallback text search using ilike"""
        try:
            # Split query into keywords
            keywords = query.lower().split()
            
            # Try searching for any of the keywords
            for keyword in keywords:
                if len(keyword) > 3:  # Only search for meaningful keywords
                    result = self.supabase.table(self.table_name)\
                        .select('content, metadata, source_url')\
                        .ilike('content', f'%{keyword}%')\
                        .limit(limit)\
                        .execute()
                    
                    if result.data:
                        logger.info(f"Text search found {len(result.data)} results for keyword: {keyword}")
                        return result.data
            
            logger.info("Text search found no results")
            return []
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    async def get_document_count(self) -> int:
        """Get total number of documents in the database"""
        try:
            result = self.supabase.table(self.table_name).select('id', count='exact').execute()
            return result.count or 0
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0

    async def clear_all_documents(self) -> bool:
        """Clear all documents from the database"""
        try:
            result = self.supabase.table(self.table_name).delete().neq('id', 0).execute()
            logger.info("Cleared all documents from database")
            return True
        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
            return False

    async def _get_recent_repository_docs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get documents from the most recently indexed repository"""
        try:
            # Get the most recent source_url (last indexed repository)
            recent_source = self.supabase.table(self.table_name)\
                .select('source_url')\
                .order('id', desc=True)\
                .limit(1)\
                .execute()
            
            if not recent_source.data:
                return []
            
            latest_source_url = recent_source.data[0]['source_url']
            logger.info(f"Most recent repository: {latest_source_url}")
            
            # Get documents from this repository
            result = self.supabase.table(self.table_name)\
                .select('content, metadata, source_url')\
                .eq('source_url', latest_source_url)\
                .limit(limit)\
                .execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Failed to get recent repository docs: {e}")
            return []
