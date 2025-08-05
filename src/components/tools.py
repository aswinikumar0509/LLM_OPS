from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from src.components.llm_instance import llm

def summarizer_fn(text: str) -> str:
    response = llm.invoke([
        {"role": "system", "content": "You are a helpful assistant that summarizes legal content with point like Case name ,Case Number,Court and bench,Legal issues,Judges,Judgment summary"},
        {"role": "user", "content": f"Summarize this:\n{text}"}
    ])
    return response.content if hasattr(response, "content") else response


summarizer_tool = Tool.from_function(
    name="Summarizer",
    description="Use this tool to summarize any text input.",
    func=summarizer_fn
)


