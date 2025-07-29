from typing import List
from dotenv import load_dotenv
from src.components.data_embedding import prepare_text_chunks_with_embeddings
from src.components.data_ingestion import load_documents_from_text_file, filter_to_minimal_docs
from src.common.logger import get_logger
from src.common.custom_exception import CustomException
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document


logger = get_logger(__name__)


def store_documents_in_pinecone(
    texts_chunk: List[Document],
    embedding,
    index_name: str = "legal-chatbot",
    dimension: int = 384,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1"
):
    """
    Stores chunked documents into a Pinecone vector store.

    Args:
        texts_chunk (List[Document]): The list of chunked documents.
        embedding: The embedding model to use.
        index_name (str): The Pinecone index name.
        dimension (int): The embedding dimension.
        metric (str): The similarity metric.
        cloud (str): Cloud provider.
        region (str): Region for serverless spec.

    Returns:
        PineconeVectorStore object or None if error
    """
    try:
        load_dotenv()

        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        hf_token = os.getenv("hf_token")

        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is missing from environment variables.")
        
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
        os.environ["hf_token"] = hf_token or ""

        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Create index if it doesn't exist
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region)
            )

        index = pc.Index(index_name)

        # Create vector store
        docsearch = PineconeVectorStore.from_documents(
            documents=texts_chunk,
            embedding=embedding,
            index_name=index_name
        )

        logger.info(f"Successfully stored {len(texts_chunk)} documents in Pinecone index '{index_name}'.")
        return docsearch
    
    except Exception as e:
        logger.error(f"Error storing documents in Pinecone: {e}")
        return None
    




def load_existing_docsearch(index_name: str = "legal-chatbot"):
    """
    Load an existing Pinecone vector index into a retrievable docsearch object.
    """
    load_dotenv()

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    hf_token = os.getenv("hf_token")

    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    os.environ["hf_token"] = hf_token or ""

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
