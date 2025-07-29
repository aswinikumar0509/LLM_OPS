import os
import sys
from src.common.logger import get_logger
from src.common.custom_exception import CustomException
from langchain_community.document_loaders import TextLoader

from langchain.schema import Document
from typing import List
from src.config.config import filepath


logger = get_logger(__name__)

## Function for loading the data

def load_documents_from_text_file(filepath: str, encoding: str = "utf-8") -> List[Document]:
    """
    Loads documents using LangChain's TextLoader with exception handling.
    """
    try:
        loader = TextLoader(filepath, encoding=encoding)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents from {filepath}")
        return docs
    except Exception as e:
        logger.error(f"Error loading documents from {filepath}: {e}")
        return []
    

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filters a list of Document objects to include only minimal metadata: 'source'.
    """
    minimal_docs = []
    try:
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            minimal_docs.append(Document(
                page_content=doc.page_content,
                metadata={"source": source}
            ))
        logger.info(f"Filtered to {len(minimal_docs)} minimal documents.")
    except Exception as e:
        logger.error(f"Error filtering documents: {e}")

    return minimal_docs

def load_documents_from_directory(directory: str, encoding: str = "utf-8") -> List[Document]:
    """
    Loads all text files from a directory and returns them as a list of Document objects.
    """
    if not os.path.isdir(directory):
        raise CustomException(f"Directory {directory} does not exist or is not a directory.")
    
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            docs.extend(load_documents_from_text_file(filepath, encoding))
    
    return filter_to_minimal_docs(docs)



