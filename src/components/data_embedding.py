from typing import List, Tuple
from src.config.config import filepath
from src.common.logger import get_logger
from src.common.custom_exception import CustomException
from src.components.data_ingestion import load_documents_from_text_file, filter_to_minimal_docs
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter




logger = get_logger(__name__)

def prepare_text_chunks_with_embeddings(minimal_docs: List[Document]) -> Tuple[List[Document], HuggingFaceEmbeddings]:
    """
    Prepares the text chunks from minimal documents and loads HuggingFace embeddings.
    
    Returns:
        texts_chunk: List of chunked Document objects
        embedding: HuggingFaceEmbeddings object for vector storage or similarity
    """
    try:
        # 1. Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        texts_chunk = text_splitter.split_documents(minimal_docs)

        # 2. Load HuggingFace Embedding model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedding = HuggingFaceEmbeddings(model_name=model_name)

        logger.info(f"Successfully prepared {len(texts_chunk)} text chunks with embeddings.")
        return texts_chunk, embedding

    except Exception as e:
        logger.error(f"Error in preparing chunks with embeddings: {e}")
        return [], None
