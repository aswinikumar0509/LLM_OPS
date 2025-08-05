import sys
import os

# Fix import issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import streamlit as st
from langchain.memory import ConversationBufferMemory
from src.components.retrival import retrieve_and_score_query
from src.components.tools import summarizer_fn  

# Initialize memory
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(return_messages=True)

# Store answer for summarization
if "retrieved_answer" not in st.session_state:
    st.session_state.retrieved_answer = ""

# Page config
st.set_page_config(page_title="Legal RAG Assistant", page_icon="⚖️", layout="wide")

# Sidebar memory viewer
with st.sidebar:
    st.markdown("### 🧠 Memory Panel")
    show_memory = st.checkbox("Show Conversation Memory", value=False)

    if show_memory:
        st.markdown("#### 📜 Previous Messages")
        messages = st.session_state.chat_memory.chat_memory.messages
        if messages:
            for msg in messages:
                role = "🧑 You" if msg.type == "human" else "🤖 Assistant"
                st.markdown(f"**{role}:** {msg.content}")
        else:
            st.info("No memory yet.")

# Main UI
st.title("⚖️ Vakki: Legal Research Assistant")
st.markdown("Ask a legal question and retrieve relevant judgments")

query = st.text_area(
    "Enter your legal question:",
    height=90,
    placeholder="Ask Vakki..."
)

col1, col2 = st.columns([1, 1])

with col1:
    retrieve_button = st.button("🔍 Retrieve Answer")

with col2:
    summarize_button = st.button("📝 Summarize Answer")

# ✅ Retrieve Answer
if retrieve_button and query.strip():
    with st.spinner("Processing your query..."):
        try:
            answer, similarity, faithfulness = retrieve_and_score_query(
                query, memory=st.session_state.chat_memory
            )
            st.session_state.retrieved_answer = answer  # ✅ Store answer

            st.success("✅ Answer Retrieved")
            st.markdown(f"**🧠 Answer:**\n\n{answer}")
            st.markdown(f"**🔁 Similarity Score (Query ↔ Answer):** `{similarity:.4f}`")
            st.markdown(f"**📚 Faithfulness Score (Context ↔ Answer):** `{faithfulness:.4f}`")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

# ✅ Summarize Retrieved Answer
elif summarize_button:
    if not st.session_state.retrieved_answer:
        st.warning("❗ No answer available to summarize. Please retrieve an answer first.")
    else:
        with st.spinner("Summarizing the retrieved answer..."):
            try:
                summary = summarizer_fn(st.session_state.retrieved_answer)
                st.success("📝 Summary Generated")
                st.markdown(f"**✂️ Summary:**\n\n{summary}")
            except Exception as e:
                st.error(f"❌ Failed to summarize: {e}")

else:
    st.info("Enter a question above and click 'Retrieve Answer' and  click 'Summarize Answer' to begin summarization of retrival content.")
