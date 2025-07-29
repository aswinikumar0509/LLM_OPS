from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.components.vector import store_documents_in_pinecone , load_existing_docsearch
from src.common.logger import get_logger
from src.common.custom_exception import CustomException
from src.components.prompt import prompt
import os


load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")


os.environ["GROQ_API_KEY"] = GROQ_API_KEY

from langchain_groq import ChatGroq

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="Llama3-8b-8192",
    temperature=1,
    top_p=0.9
)
docsearch = load_existing_docsearch("legal-chatbot2")
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":5})

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)