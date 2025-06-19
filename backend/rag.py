import os
import asyncio
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import tempfile
import shutil
import json

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Supabase imports
from supabase import create_client, Client

# Other imports
import git
import requests
from dotenv import load_dotenv
import numpy as np

load_dotenv()

logger = logging.getLogger(__name__)

class RAGSystem:
    """
    A complete RAG system that can:
    1. Load and index GitHub repositories
    2. Store embeddings in Supabase database
    3. Answer questions using Gemini 2.0 Flash
    """
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY") 
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not all([self.supabase_url, self.supabase_key, self.gemini_api_key]):
            raise ValueError("Missing required environment variables: SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY")
        
        # Initialize Supabase client
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.gemini_api_key
        )
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=self.gemini_api_key,
            temperature=0.1,
            max_tokens=1000
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Table name for storing documents
        self.table_name = "documents"
        
    async def _setup_vector_table(self):
        """Set up the documents table in Supabase"""
        try:
            # First, try to create the table if it doesn't exist
            # This is a simplified approach using Supabase's REST API
            result = self.supabase.table(self.table_name).select("id").limit(1).execute()
            logger.info("Documents table already exists")
            return True
        except Exception as e:
            logger.info(f"Table might not exist, creating it: {e}")
            # For now, we'll assume the table exists or will be created manually
            # In a production app, you'd set this up through Supabase dashboard
            return True
    
    async def index_github_repo(self, github_url: str) -> Dict[str, Any]:
        """
        Clone a GitHub repository and index its contents
        """
        temp_dir = None
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Cloning repository to: {temp_dir}")
            
            # Clone the repository
            repo = git.Repo.clone_from(github_url, temp_dir)
            logger.info(f"Successfully cloned repository: {github_url}")
            
            # Load documents from the cloned repository
            documents = self._load_documents_from_path(temp_dir)
            logger.info(f"Loaded {len(documents)} documents from repository")
            
            if not documents:
                return {"documents_processed": 0, "chunks_created": 0}
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from documents")
            
            # Store chunks in Supabase
            await self._store_chunks(chunks, github_url)
            
            return {
                "documents_processed": len(documents),
                "chunks_created": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error indexing GitHub repository: {e}")
            raise e
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary directory")
    
    def _load_documents_from_path(self, path: str) -> List[Document]:
        """
        Load documents from a local path, focusing on text and markdown files
        """
        documents = []
        path_obj = Path(path)
        
        # File extensions to process
        text_extensions = {'.md', '.txt', '.py', '.js', '.ts', '.json', '.yaml', '.yml', '.rst'}
        
        for file_path in path_obj.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in text_extensions:
                try:
                    # Skip large files (>1MB)
                    if file_path.stat().st_size > 1024 * 1024:
                        continue
                    
                    # Skip hidden files and directories
                    if any(part.startswith('.') for part in file_path.parts):
                        continue
                    
                    # Read file content directly
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Create document
                    doc = Document(
                        page_content=content,
                        metadata={
                            'file_path': str(file_path.relative_to(path_obj)),
                            'file_type': file_path.suffix.lower(),
                            'file_name': file_path.name
                        }
                    )
                    
                    documents.append(doc)
                    logger.debug(f"Loaded document: {file_path}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load document {file_path}: {e}")
                    continue
        
        return documents
    
    async def _store_chunks(self, chunks: List[Document], source_url: str):
        """
        Store document chunks in Supabase database
        """
        try:
            # Setup table
            await self._setup_vector_table()
            
            # Clear existing data for this source
            self.supabase.table(self.table_name).delete().eq('source_url', source_url).execute()
            
            # Prepare data for insertion
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embeddings.embed_documents(texts)
            
            # Prepare records for insertion
            records = []
            for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
                record = {
                    'content': text,
                    'metadata': json.dumps(metadata),
                    'embedding': embedding,
                    'source_url': source_url,
                    'chunk_index': i
                }
                records.append(record)
            
            # Insert in batches to avoid timeouts
            batch_size = 10
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                result = self.supabase.table(self.table_name).insert(batch).execute()
                logger.info(f"Inserted batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size}")
            
            logger.info("Successfully stored chunks in Supabase")
            
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            raise e
    
    async def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using RAG: retrieve relevant chunks and generate answer
        """
        try:
            # Generate embedding for the question
            question_embedding = self.embeddings.embed_query(question)
            
            # Search for relevant chunks using cosine similarity
            # Note: This is a simplified approach. In production, you'd use pgvector
            chunks = await self._search_similar_chunks(question_embedding)
            
            if not chunks:
                return {
                    "answer": "I don't have enough information to answer that question. Please make sure you've indexed a repository first.",
                    "sources": []
                }
            
            # Extract context from search results
            context_chunks = []
            sources = []
            
            for chunk in chunks:
                context_chunks.append(chunk['content'])
                metadata = json.loads(chunk.get('metadata', '{}'))
                sources.append(metadata.get('file_path', 'Unknown'))
            
            context = "\n\n".join(context_chunks)
            
            # Create prompt for Gemini
            prompt_template = ChatPromptTemplate.from_template("""
You are a helpful AI assistant that answers questions based on the provided context.
Use only the information from the context to answer the question.
If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {question}

Answer:""")
            
            # Generate answer
            chain = prompt_template | self.llm | StrOutputParser()
            
            answer = await chain.ainvoke({
                "context": context,
                "question": question
            })
            
            return {
                "answer": answer.strip(),
                "sources": list(set(sources))  # Remove duplicates
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise e
    
    async def _search_similar_chunks(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        """
        Search for similar chunks using simple cosine similarity
        Note: This is a simplified approach. In production, use pgvector extension
        """
        try:
            # Get all chunks from database
            result = self.supabase.table(self.table_name).select("*").execute()
            
            if not result.data:
                return []
            
            # Calculate cosine similarity for each chunk
            similarities = []
            for chunk in result.data:
                chunk_embedding = chunk['embedding']
                if chunk_embedding:
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                    similarities.append((similarity, chunk))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [chunk for _, chunk in similarities[:limit]]
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except:
            return 0.0
