import os
from dotenv import load_dotenv
from src.common.logger import get_logger
from src.common.custom_exception import CustomException
from src.components.data_ingestion import load_documents_from_text_file, filter_to_minimal_docs
from src.components.data_embedding import prepare_text_chunks_with_embeddings
from src.components.vector import store_documents_in_pinecone
from src.config.config import filepath
import sys
import os

logger = get_logger(__name__)

def run_llmops_data_pipeline(file_path: str, index_name: str = "legal-chatbot2"):
    """
    Runs the complete LLMOps pipeline from data ingestion to Pinecone vector storage.
    """
    try:
        logger.info("🚀 Starting the LLMOps data pipeline...")

        # Step 1: Load and filter documents
        docs = load_documents_from_text_file(file_path)
        if not docs:
            raise CustomException("No documents loaded. Please check your input file.")

        minimal_docs = filter_to_minimal_docs(docs)

        # Step 2: Text chunking and embedding
        texts_chunk, embedding = prepare_text_chunks_with_embeddings(minimal_docs)
        if not texts_chunk or embedding is None:
            raise CustomException("Text chunking or embedding failed.")

        # Step 3: Store in Pinecone
        docsearch = store_documents_in_pinecone(
            texts_chunk=texts_chunk,
            embedding=embedding,
            index_name=index_name
        )

        if docsearch:
            logger.info("✅ Pipeline completed and documents stored in Pinecone.")
        else:
            logger.warning("⚠️ Pipeline completed but failed to store documents in Pinecone.")

    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")


# 👇 Add this to allow direct execution
if __name__ == "__main__":
    load_dotenv()  

    
    file_path = "reaearch\combined_output2.txt" # from src/config/config.py
    index_name = "legal-chatbot2"

    run_llmops_data_pipeline(file_path, index_name)
