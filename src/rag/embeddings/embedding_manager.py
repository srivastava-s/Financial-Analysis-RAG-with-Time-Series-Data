"""
Embedding manager for the Financial Analysis RAG System.
"""

import openai
import numpy as np
from typing import List, Optional, Dict, Any, Union
import logging
from sentence_transformers import SentenceTransformer
import hashlib
import json
import pickle
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...models import DocumentChunk, NewsArticle, FinancialDocument
from ...config import config

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages text embeddings for the RAG system."""
    
    def __init__(self, model_name: str = "openai"):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.openai_client = None
        self.sentence_transformer = None
        self.cache_dir = Path(config.EMBEDDINGS_DIR) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            if self.model_name == "openai":
                if config.OPENAI_API_KEY:
                    self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
                    logger.info("Initialized OpenAI embedding model")
                else:
                    logger.warning("OpenAI API key not found, falling back to sentence transformers")
                    self.model_name = "sentence-transformers"
            
            if self.model_name == "sentence-transformers":
                # Use a financial domain-specific model if available
                model_path = "all-MiniLM-L6-v2"  # Good balance of speed and quality
                self.sentence_transformer = SentenceTransformer(model_path)
                logger.info(f"Initialized SentenceTransformer model: {model_path}")
            
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
    
    def get_embedding(self, text: str, use_cache: bool = True) -> Optional[List[float]]:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use caching
            
        Returns:
            Embedding vector or None
        """
        try:
            # Check cache first
            if use_cache:
                cached_embedding = self._get_cached_embedding(text)
                if cached_embedding is not None:
                    return cached_embedding
            
            # Generate embedding
            if self.model_name == "openai":
                embedding = self._get_openai_embedding(text)
            else:
                embedding = self._get_sentence_transformer_embedding(text)
            
            # Cache the embedding
            if use_cache and embedding is not None:
                self._cache_embedding(text, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def get_embeddings_batch(
        self, 
        texts: List[str], 
        use_cache: bool = True,
        batch_size: int = 100
    ) -> List[Optional[List[float]]]:
        """
        Get embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use caching
            batch_size: Size of batches for processing
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._process_batch(batch_texts, use_cache)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _process_batch(
        self, 
        texts: List[str], 
        use_cache: bool
    ) -> List[Optional[List[float]]]:
        """Process a batch of texts for embedding."""
        embeddings = []
        
        # Check cache first
        if use_cache:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cached_embedding = self._get_cached_embedding(text)
                if cached_embedding is not None:
                    cached_embeddings.append((i, cached_embedding))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                if self.model_name == "openai":
                    new_embeddings = self._get_openai_embeddings_batch(uncached_texts)
                else:
                    new_embeddings = self._get_sentence_transformer_embeddings_batch(uncached_texts)
                
                # Cache new embeddings
                for text, embedding in zip(uncached_texts, new_embeddings):
                    if embedding is not None:
                        self._cache_embedding(text, embedding)
                
                # Combine cached and new embeddings
                all_embeddings = [None] * len(texts)
                for i, embedding in cached_embeddings:
                    all_embeddings[i] = embedding
                for i, embedding in zip(uncached_indices, new_embeddings):
                    all_embeddings[i] = embedding
                
                embeddings = all_embeddings
            else:
                # All embeddings were cached
                all_embeddings = [None] * len(texts)
                for i, embedding in cached_embeddings:
                    all_embeddings[i] = embedding
                embeddings = all_embeddings
        else:
            # No caching
            if self.model_name == "openai":
                embeddings = self._get_openai_embeddings_batch(texts)
            else:
                embeddings = self._get_sentence_transformer_embeddings_batch(texts)
        
        return embeddings
    
    def _get_openai_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding using OpenAI API."""
        try:
            response = self.openai_client.embeddings.create(
                model=config.OPENAI_EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {e}")
            return None
    
    def _get_openai_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Get embeddings for multiple texts using OpenAI API."""
        try:
            response = self.openai_client.embeddings.create(
                model=config.OPENAI_EMBEDDING_MODEL,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error getting OpenAI embeddings batch: {e}")
            return [None] * len(texts)
    
    def _get_sentence_transformer_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding using SentenceTransformer."""
        try:
            embedding = self.sentence_transformer.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error getting SentenceTransformer embedding: {e}")
            return None
    
    def _get_sentence_transformer_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Get embeddings for multiple texts using SentenceTransformer."""
        try:
            embeddings = self.sentence_transformer.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error getting SentenceTransformer embeddings batch: {e}")
            return [None] * len(texts)
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        try:
            text_hash = self._hash_text(text)
            cache_file = self.cache_dir / f"{text_hash}.pkl"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    return cached_data['embedding']
            
            return None
        except Exception as e:
            logger.warning(f"Error reading cached embedding: {e}")
            return None
    
    def _cache_embedding(self, text: str, embedding: List[float]):
        """Cache embedding for text."""
        try:
            text_hash = self._hash_text(text)
            cache_file = self.cache_dir / f"{text_hash}.pkl"
            
            cached_data = {
                'text': text,
                'embedding': embedding,
                'model': self.model_name,
                'timestamp': time.time()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
                
        except Exception as e:
            logger.warning(f"Error caching embedding: {e}")
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def embed_documents(
        self, 
        documents: List[Union[DocumentChunk, NewsArticle, FinancialDocument]],
        use_cache: bool = True
    ) -> List[DocumentChunk]:
        """
        Embed a list of documents.
        
        Args:
            documents: List of documents to embed
            use_cache: Whether to use caching
            
        Returns:
            List of DocumentChunk objects with embeddings
        """
        embedded_chunks = []
        
        for doc in documents:
            try:
                # Extract text content
                if isinstance(doc, DocumentChunk):
                    text = doc.content
                    chunk = doc
                elif isinstance(doc, NewsArticle):
                    text = f"{doc.title} {doc.content}"
                    chunk = DocumentChunk(
                        id=doc.id,
                        document_id=doc.id,
                        content=text,
                        chunk_index=0,
                        start_char=0,
                        end_char=len(text),
                        metadata={
                            'type': 'news_article',
                            'title': doc.title,
                            'source': doc.source,
                            'published_at': doc.published_at.isoformat(),
                            'sentiment_score': doc.sentiment_score,
                            'sentiment_type': doc.sentiment_type.value if doc.sentiment_type else None,
                            'keywords': doc.keywords,
                            'relevance_score': doc.relevance_score
                        }
                    )
                elif isinstance(doc, FinancialDocument):
                    text = f"{doc.title} {doc.content}"
                    chunk = DocumentChunk(
                        id=doc.id,
                        document_id=doc.id,
                        content=text,
                        chunk_index=0,
                        start_char=0,
                        end_char=len(text),
                        metadata={
                            'type': 'financial_document',
                            'title': doc.title,
                            'document_type': doc.document_type.value,
                            'company_symbol': doc.company_symbol,
                            'filing_date': doc.filing_date.isoformat() if doc.filing_date else None,
                            'source': doc.source.value
                        }
                    )
                else:
                    continue
                
                # Get embedding
                embedding = self.get_embedding(text, use_cache)
                if embedding is not None:
                    chunk.embedding = embedding
                    embedded_chunks.append(chunk)
                
            except Exception as e:
                logger.warning(f"Error embedding document {doc.id}: {e}")
                continue
        
        return embedded_chunks
    
    def chunk_document(
        self, 
        document: Union[NewsArticle, FinancialDocument],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[DocumentChunk]:
        """
        Chunk a document into smaller pieces for embedding.
        
        Args:
            document: Document to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of DocumentChunk objects
        """
        if chunk_size is None:
            chunk_size = config.CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = config.CHUNK_OVERLAP
        
        chunks = []
        
        try:
            # Extract text content
            if isinstance(document, NewsArticle):
                text = f"{document.title} {document.content}"
                doc_id = document.id
                metadata = {
                    'type': 'news_article',
                    'title': document.title,
                    'source': document.source,
                    'published_at': document.published_at.isoformat(),
                    'sentiment_score': document.sentiment_score,
                    'sentiment_type': document.sentiment_type.value if document.sentiment_type else None,
                    'keywords': document.keywords,
                    'relevance_score': document.relevance_score
                }
            elif isinstance(document, FinancialDocument):
                text = f"{document.title} {document.content}"
                doc_id = document.id
                metadata = {
                    'type': 'financial_document',
                    'title': document.title,
                    'document_type': document.document_type.value,
                    'company_symbol': document.company_symbol,
                    'filing_date': document.filing_date.isoformat() if document.filing_date else None,
                    'source': document.source.value
                }
            else:
                return chunks
            
            # Split text into chunks
            text_length = len(text)
            start = 0
            chunk_index = 0
            
            while start < text_length:
                end = min(start + chunk_size, text_length)
                
                # Adjust end to not break words
                if end < text_length:
                    last_space = text.rfind(' ', start, end)
                    if last_space > start:
                        end = last_space
                
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunk = DocumentChunk(
                        id=f"{doc_id}_chunk_{chunk_index}",
                        document_id=doc_id,
                        content=chunk_text,
                        chunk_index=chunk_index,
                        start_char=start,
                        end_char=end,
                        metadata=metadata.copy()
                    )
                    chunks.append(chunk)
                
                start = end - chunk_overlap
                chunk_index += 1
                
                if start >= text_length:
                    break
        
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
        
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        if self.model_name == "openai":
            return config.EMBEDDING_DIMENSION
        elif self.sentence_transformer:
            return self.sentence_transformer.get_sentence_embedding_dimension()
        else:
            return 384  # Default for all-MiniLM-L6-v2
    
    def clear_cache(self):
        """Clear the embedding cache."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Embedding cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache."""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                'cache_files': len(cache_files),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'cache_dir': str(self.cache_dir)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
