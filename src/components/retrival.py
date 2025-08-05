import numpy as np
from typing import Tuple, List
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory  # Optional for typing

from src.common.logger import get_logger
from src.components.llm import rag_chain

logger = get_logger(__name__)

def retrieve_and_score_query(
    query: str,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
    memory: ConversationBufferMemory = None  # ✅ Accept memory
) -> Tuple[str, float, float]:
    """
    Execute a RAG query and compute similarity + faithfulness scores.

    Args:
        query (str): User query.
        embedding_model_name (str): HuggingFace model to use.
        top_k (int): Number of top docs to consider for faithfulness.
        memory (ConversationBufferMemory): Optional memory object.

    Returns:
        Tuple:
            - answer (str)
            - similarity score (query ↔ answer)
            - faithfulness score (context ↔ answer)
    """
    try:
        logger.info(f"🔍 Query: {query}")
        embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
        if memory:
            history = memory.buffer
            combined_query = f"{history}\nUser: {query}"
            logger.debug(f"📨 Combined input with memory: {combined_query}")
        else:
            combined_query = query

        # Step 1: Call RAG chain
        response = rag_chain.invoke({"input": query})
        answer = response.get("answer", "")
        context_docs = response.get("context", [])

        # Step 2: Embed elements
        query_embedding = embedding.embed_query(query)
        answer_embedding = embedding.embed_query(answer)

        context_texts = [doc.page_content for doc in context_docs[:top_k]]
        context_embeddings = embedding.embed_documents(context_texts)
        context_embedding_avg = np.mean(context_embeddings, axis=0)

        # Step 3: Similarity Calculations
        similarity = cosine_similarity([query_embedding], [answer_embedding])[0][0]
        faithfulness = cosine_similarity([context_embedding_avg], [answer_embedding])[0][0]

        logger.info(f"✅ Similarity (query ↔ answer): {similarity:.4f}")
        logger.info(f"✅ Faithfulness (context ↔ answer): {faithfulness:.4f}")

        # Step 4: Add to memory if provided
        if memory:
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(answer)
            logger.debug(f"🧠 Memory updated with query and answer.")

        return answer, similarity, faithfulness

    except Exception as e:
        logger.error(f"❌ Retrieval failed: {e}")
        raise e
