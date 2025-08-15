"""
Vector store manager using ChromaDB for the Financial Analysis RAG System.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Optional, Dict, Any, Union
import logging
from datetime import datetime, timedelta
import json
import uuid
from pathlib import Path

from ...models import DocumentChunk, NewsArticle, FinancialDocument, RetrievalResult
from ...config import config

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store manager using ChromaDB."""
    
    def __init__(self, collection_name: str = "financial_documents"):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        self._initialize_client()
        self._initialize_collection()
    
    def _initialize_client(self):
        """Initialize ChromaDB client."""
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=config.CHROMA_DB_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"Initialized ChromaDB client at {config.CHROMA_DB_PATH}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {e}")
            raise
    
    def _initialize_collection(self):
        """Initialize or get the collection."""
        try:
            # Check if collection exists
            collections = self.client.list_collections()
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if collection_exists:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(f"Connected to existing collection: {self.collection_name}")
            else:
                # Create new collection
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "description": "Financial documents and news articles for RAG system",
                        "created_at": datetime.now().isoformat()
                    }
                )
                logger.info(f"Created new collection: {self.collection_name}")
        
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
    
    def add_documents(
        self, 
        documents: List[Union[DocumentChunk, NewsArticle, FinancialDocument]],
        batch_size: int = 100
    ) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            batch_size: Size of batches for processing
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self._add_batch(batch)
            
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    def _add_batch(self, documents: List[Union[DocumentChunk, NewsArticle, FinancialDocument]]):
        """Add a batch of documents to the collection."""
        try:
            ids = []
            embeddings = []
            metadatas = []
            documents_text = []
            
            for doc in documents:
                # Extract document information
                if isinstance(doc, DocumentChunk):
                    doc_id = doc.id
                    embedding = doc.embedding
                    metadata = doc.metadata
                    text = doc.content
                elif isinstance(doc, NewsArticle):
                    doc_id = doc.id
                    # Generate embedding if not already present
                    from ...rag.embeddings.embedding_manager import EmbeddingManager
                    embedding_manager = EmbeddingManager()
                    embedding = embedding_manager.get_embedding(f"{doc.title} {doc.content}")
                    metadata = {
                        'type': 'news_article',
                        'title': doc.title,
                        'source': doc.source,
                        'published_at': doc.published_at.isoformat(),
                        'sentiment_score': doc.sentiment_score,
                        'sentiment_type': doc.sentiment_type.value if doc.sentiment_type else None,
                        'keywords': json.dumps(doc.keywords),
                        'relevance_score': doc.relevance_score
                    }
                    text = f"{doc.title} {doc.content}"
                elif isinstance(doc, FinancialDocument):
                    doc_id = doc.id
                    # Generate embedding if not already present
                    from ...rag.embeddings.embedding_manager import EmbeddingManager
                    embedding_manager = EmbeddingManager()
                    embedding = embedding_manager.get_embedding(f"{doc.title} {doc.content}")
                    metadata = {
                        'type': 'financial_document',
                        'title': doc.title,
                        'document_type': doc.document_type.value,
                        'company_symbol': doc.company_symbol,
                        'filing_date': doc.filing_date.isoformat() if doc.filing_date else None,
                        'source': doc.source.value
                    }
                    text = f"{doc.title} {doc.content}"
                else:
                    continue
                
                # Skip if no embedding
                if embedding is None:
                    logger.warning(f"No embedding for document {doc_id}")
                    continue
                
                ids.append(doc_id)
                embeddings.append(embedding)
                metadatas.append(metadata)
                documents_text.append(text)
            
            # Add to collection
            if ids:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents_text
                )
        
        except Exception as e:
            logger.error(f"Error adding batch to collection: {e}")
            raise
    
    def search(
        self, 
        query: str,
        query_embedding: Optional[List[float]] = None,
        n_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> RetrievalResult:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            query_embedding: Pre-computed query embedding
            n_results: Number of results to return
            filter_metadata: Metadata filters
            include_metadata: Whether to include metadata in results
            
        Returns:
            RetrievalResult object
        """
        try:
            # Generate query embedding if not provided
            if query_embedding is None:
                from ...rag.embeddings.embedding_manager import EmbeddingManager
                embedding_manager = EmbeddingManager()
                query_embedding = embedding_manager.get_embedding(query)
                
                if query_embedding is None:
                    logger.error("Failed to generate query embedding")
                    return RetrievalResult(
                        query=query,
                        documents=[],
                        scores=[],
                        metadata={}
                    )
            
            # Prepare where clause for filtering
            where_clause = None
            if filter_metadata:
                where_clause = self._build_where_clause(filter_metadata)
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Convert results to DocumentChunk objects
            documents = []
            scores = []
            
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    try:
                        # Create DocumentChunk from results
                        chunk = DocumentChunk(
                            id=doc_id,
                            document_id=doc_id,
                            content=results['documents'][0][i],
                            chunk_index=0,
                            start_char=0,
                            end_char=len(results['documents'][0][i]),
                            metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                        )
                        
                        documents.append(chunk)
                        
                        # Convert distance to similarity score
                        distance = results['distances'][0][i]
                        similarity_score = 1.0 / (1.0 + distance)  # Convert distance to similarity
                        scores.append(similarity_score)
                        
                    except Exception as e:
                        logger.warning(f"Error processing search result {i}: {e}")
                        continue
            
            return RetrievalResult(
                query=query,
                documents=documents,
                scores=scores,
                metadata={
                    'total_results': len(documents),
                    'search_timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return RetrievalResult(
                query=query,
                documents=[],
                scores=[],
                metadata={'error': str(e)}
            )
    
    def search_by_company(
        self, 
        company_symbol: str,
        query: str,
        n_results: int = 10
    ) -> RetrievalResult:
        """
        Search for documents related to a specific company.
        
        Args:
            company_symbol: Company stock symbol
            query: Search query
            n_results: Number of results to return
            
        Returns:
            RetrievalResult object
        """
        filter_metadata = {
            'company_symbol': company_symbol
        }
        
        return self.search(query, n_results=n_results, filter_metadata=filter_metadata)
    
    def search_by_date_range(
        self, 
        query: str,
        start_date: datetime,
        end_date: datetime,
        n_results: int = 10
    ) -> RetrievalResult:
        """
        Search for documents within a date range.
        
        Args:
            query: Search query
            start_date: Start date
            end_date: End date
            n_results: Number of results to return
            
        Returns:
            RetrievalResult object
        """
        filter_metadata = {
            'published_at': {
                '$gte': start_date.isoformat(),
                '$lte': end_date.isoformat()
            }
        }
        
        return self.search(query, n_results=n_results, filter_metadata=filter_metadata)
    
    def search_by_document_type(
        self, 
        query: str,
        document_type: str,
        n_results: int = 10
    ) -> RetrievalResult:
        """
        Search for documents of a specific type.
        
        Args:
            query: Search query
            document_type: Type of document
            n_results: Number of results to return
            
        Returns:
            RetrievalResult object
        """
        filter_metadata = {
            'type': document_type
        }
        
        return self.search(query, n_results=n_results, filter_metadata=filter_metadata)
    
    def _build_where_clause(self, filter_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build ChromaDB where clause from filter metadata.
        
        Args:
            filter_metadata: Metadata filters
            
        Returns:
            ChromaDB where clause
        """
        where_clause = {}
        
        for key, value in filter_metadata.items():
            if isinstance(value, dict) and ('$gte' in value or '$lte' in value):
                # Range query
                where_clause[key] = value
            else:
                # Exact match
                where_clause[key] = value
        
        return where_clause
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample documents to analyze metadata
            sample_results = self.collection.peek(limit=100)
            
            # Analyze document types
            doc_types = {}
            if sample_results['metadatas']:
                for metadata in sample_results['metadatas']:
                    doc_type = metadata.get('type', 'unknown')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            return {
                'total_documents': count,
                'document_types': doc_types,
                'collection_name': self.collection_name,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents from the collection.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from collection")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def update_document(
        self, 
        document_id: str, 
        new_content: str, 
        new_embedding: Optional[List[float]] = None
    ) -> bool:
        """
        Update a document in the collection.
        
        Args:
            document_id: Document ID to update
            new_content: New document content
            new_embedding: New document embedding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate new embedding if not provided
            if new_embedding is None:
                from ...rag.embeddings.embedding_manager import EmbeddingManager
                embedding_manager = EmbeddingManager()
                new_embedding = embedding_manager.get_embedding(new_content)
                
                if new_embedding is None:
                    logger.error("Failed to generate new embedding")
                    return False
            
            # Update the document
            self.collection.update(
                ids=[document_id],
                embeddings=[new_embedding],
                documents=[new_content]
            )
            
            logger.info(f"Updated document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(where={})
            logger.info("Cleared all documents from collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def backup_collection(self, backup_path: str) -> bool:
        """
        Backup the collection to a file.
        
        Args:
            backup_path: Path to save the backup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all documents
            results = self.collection.get()
            
            # Save to file
            backup_data = {
                'collection_name': self.collection_name,
                'documents': results,
                'backup_timestamp': datetime.now().isoformat()
            }
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"Backed up collection to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up collection: {e}")
            return False
    
    def restore_collection(self, backup_path: str) -> bool:
        """
        Restore the collection from a backup file.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load backup data
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            # Clear current collection
            self.clear_collection()
            
            # Restore documents
            if 'documents' in backup_data:
                documents = backup_data['documents']
                if documents['ids']:
                    self.collection.add(
                        ids=documents['ids'],
                        embeddings=documents['embeddings'],
                        metadatas=documents['metadatas'],
                        documents=documents['documents']
                    )
            
            logger.info(f"Restored collection from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring collection: {e}")
            return False
