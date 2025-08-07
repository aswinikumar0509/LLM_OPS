from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.components.vector import store_documents_in_pinecone , load_existing_docsearch
from src.common.logger import get_logger
from src.common.custom_exception import CustomException
from src.components.prompt import prompt
from src.components.data_ingestion import load_documents_from_text_file, filter_to_minimal_docs
from src.components.data_embedding import prepare_text_chunks_with_embeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import os


load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")


os.environ["GROQ_API_KEY"] = GROQ_API_KEY

from langchain_groq import ChatGroq

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="Llama3-8b-8192",
    temperature=0.8,
    top_p=0.9
)

file_path = "reaearch\combined_output2.txt"
docs = load_documents_from_text_file(file_path)
if not docs:
    raise CustomException("No documents loaded. Please check your input file.")

minimal_docs = filter_to_minimal_docs(docs)

texts_chunk, embedding = prepare_text_chunks_with_embeddings(minimal_docs)
if not texts_chunk or embedding is None:
    raise CustomException("Text chunking or embedding failed.")



bm25_retriever = BM25Retriever.from_documents(texts_chunk)
bm25_retriever.k = 5

docsearch = load_existing_docsearch("legal-chatbot2")
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":5})

hybrid_retriever = EnsembleRetriever(
    retrievers=[retriever, bm25_retriever],
    weights=[0.5, 0.5]  # Tune based on testing
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(hybrid_retriever, question_answer_chain)