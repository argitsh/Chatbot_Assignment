import streamlit as st
from src.pipeline import rag_pipeline

st.set_page_config(page_title="Colab RAG Chatbot", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []

st.title("RAG Chatbot (StableLM-Zephyr)")

for role, msg in st.session_state.messages:
    st.markdown(f"**{role.capitalize()}:** {msg}")

user_query = st.text_input("Ask a question about the document:")

if st.button("Send") and user_query:
    st.session_state.messages.append(("user", user_query))

    response_stream, sources = rag_pipeline(user_query)
    st.session_state.sources = sources

    st.markdown("**Assistant:**")
    response_placeholder = st.empty()
    partial_text = ""
    for partial_text in response_stream:
        response_placeholder.markdown(partial_text)
    st.session_state.messages.append(("assistant", partial_text))

if st.session_state.sources:
    st.subheader("Sources used")
    for i, src in enumerate(st.session_state.sources):
        st.markdown(f"**[{i+1}]** {src['text'][:400]}...")
        st.caption(f"Distance: {src['distance']:.4f}")
