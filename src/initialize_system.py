"""
System initialization script for Financial Analysis RAG System.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import config
from rag.generation.rag_system import FinancialRAGSystem
from data.storage.vector_store import VectorStore
from rag.embeddings.embedding_manager import EmbeddingManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_system():
    """Initialize the Financial Analysis RAG System."""
    
    print("🚀 Initializing Financial Analysis RAG System...")
    
    try:
        # Step 1: Validate configuration
        print("📋 Validating configuration...")
        if not config.validate_config():
            print("❌ Configuration validation failed!")
            print("Please check your environment variables and configuration.")
            return False
        
        # Step 2: Create directories
        print("📁 Creating directories...")
        config.create_directories()
        print("✅ Directories created successfully")
        
        # Step 3: Initialize components
        print("🔧 Initializing system components...")
        
        # Initialize embedding manager
        print("  - Initializing embedding manager...")
        embedding_manager = EmbeddingManager()
        print(f"    ✅ Embedding dimension: {embedding_manager.get_embedding_dimension()}")
        
        # Initialize vector store
        print("  - Initializing vector store...")
        vector_store = VectorStore()
        stats = vector_store.get_collection_stats()
        print(f"    ✅ Vector store initialized with {stats.get('total_documents', 0)} documents")
        
        # Initialize RAG system
        print("  - Initializing RAG system...")
        rag_system = FinancialRAGSystem()
        print("    ✅ RAG system initialized successfully")
        
        # Step 4: Test system components
        print("🧪 Testing system components...")
        
        # Test embedding generation
        test_text = "This is a test for the financial analysis system."
        embedding = embedding_manager.get_embedding(test_text)
        if embedding:
            print("    ✅ Embedding generation working")
        else:
            print("    ⚠️ Embedding generation failed (using fallback)")
        
        # Test vector store
        try:
            test_results = vector_store.search("test query", n_results=1)
            print("    ✅ Vector store search working")
        except Exception as e:
            print(f"    ⚠️ Vector store search failed: {e}")
        
        # Step 5: System status
        print("\n📊 System Status:")
        print(f"  - Configuration: {'✅ Valid' if config.validate_config() else '❌ Invalid'}")
        print(f"  - Embedding Manager: ✅ Ready")
        print(f"  - Vector Store: ✅ Ready")
        print(f"  - RAG System: ✅ Ready")
        print(f"  - Data Directory: {config.DATA_DIR}")
        print(f"  - ChromaDB Path: {config.CHROMA_DB_PATH}")
        
        # Step 6: Cache statistics
        cache_stats = embedding_manager.get_cache_stats()
        if cache_stats:
            print(f"\n💾 Cache Statistics:")
            print(f"  - Cache files: {cache_stats.get('cache_files', 0)}")
            print(f"  - Cache size: {cache_stats.get('total_size_mb', 0):.2f} MB")
        
        print("\n🎉 System initialization completed successfully!")
        print("\n📝 Next steps:")
        print("  1. Set up your API keys in the .env file")
        print("  2. Run the Streamlit app: streamlit run src/app.py")
        print("  3. Or use the CLI: python src/cli.py --help")
        
        return True
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        print(f"❌ System initialization failed: {e}")
        return False

def check_system_health():
    """Check the health of the system components."""
    
    print("🏥 Checking system health...")
    
    try:
        # Check configuration
        config_valid = config.validate_config()
        print(f"  - Configuration: {'✅ Valid' if config_valid else '❌ Invalid'}")
        
        # Check directories
        dirs_exist = all([
            config.DATA_DIR.exists(),
            config.RAW_DATA_DIR.exists(),
            config.PROCESSED_DATA_DIR.exists(),
            config.EMBEDDINGS_DIR.exists()
        ])
        print(f"  - Directories: {'✅ Exist' if dirs_exist else '❌ Missing'}")
        
        # Check components
        try:
            embedding_manager = EmbeddingManager()
            print("  - Embedding Manager: ✅ Healthy")
        except Exception as e:
            print(f"  - Embedding Manager: ❌ Error - {e}")
        
        try:
            vector_store = VectorStore()
            stats = vector_store.get_collection_stats()
            print(f"  - Vector Store: ✅ Healthy ({stats.get('total_documents', 0)} documents)")
        except Exception as e:
            print(f"  - Vector Store: ❌ Error - {e}")
        
        try:
            rag_system = FinancialRAGSystem()
            print("  - RAG System: ✅ Healthy")
        except Exception as e:
            print(f"  - RAG System: ❌ Error - {e}")
        
        print("\n✅ System health check completed!")
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")

def clear_system_data():
    """Clear all system data (use with caution)."""
    
    response = input("⚠️ This will clear all system data. Are you sure? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    try:
        print("🗑️ Clearing system data...")
        
        # Clear vector store
        vector_store = VectorStore()
        vector_store.clear_collection()
        print("  - Vector store cleared")
        
        # Clear embedding cache
        embedding_manager = EmbeddingManager()
        embedding_manager.clear_cache()
        print("  - Embedding cache cleared")
        
        # Clear data directories
        import shutil
        for data_dir in [config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR, config.EMBEDDINGS_DIR]:
            if data_dir.exists():
                shutil.rmtree(data_dir)
                data_dir.mkdir(parents=True, exist_ok=True)
        
        print("  - Data directories cleared")
        print("✅ System data cleared successfully!")
        
    except Exception as e:
        print(f"❌ Failed to clear system data: {e}")

def main():
    """Main function for system initialization."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Financial Analysis RAG System Initialization")
    parser.add_argument("--action", choices=["init", "health", "clear"], default="init",
                       help="Action to perform (init, health, clear)")
    
    args = parser.parse_args()
    
    if args.action == "init":
        initialize_system()
    elif args.action == "health":
        check_system_health()
    elif args.action == "clear":
        clear_system_data()

if __name__ == "__main__":
    main()
