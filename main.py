import sys
import os

# Fix import issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
import streamlit as st
from src.components.retrival import retrieve_and_score_query

st.set_page_config(page_title="Legal RAG Assistant", page_icon="⚖️", layout="centered")

st.title("⚖️ Legal Research Assistant")
st.markdown("Ask a legal question and retrieve relevant judgments using RAG (LLM + Pinecone).")

# Input from user
query = st.text_area("Enter your legal question:", height=90, placeholder="e.g., Provide me top 5 judgments for a rape case...")

# Submit
if st.button("🔍 Retrieve Answer") and query.strip():
    with st.spinner("Processing your query..."):
        try:
            answer, similarity, faithfulness = retrieve_and_score_query(query)

            # Display results
            st.success("✅ Answer Retrieved")
            st.markdown(f"**🧠 Answer:** {answer}")
            st.markdown(f"**🔁 Similarity Score (Query ↔ Answer):** `{similarity:.4f}`")
            st.markdown(f"**📚 Faithfulness Score (Context ↔ Answer):** `{faithfulness:.4f}`")


        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
else:
    st.info("Enter a question above and click 'Retrieve Answer' to begin.")
