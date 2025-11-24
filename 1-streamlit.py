# python 3.13
# streamlit run 1-streamlit.py

from typing import List
from pydantic import BaseModel
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents.base import Document
from langgraph.graph import StateGraph, START, END

# Streamlit config
st.title("🦜: Agentic RAG Chatbot LangGraph")
st.sidebar.title("Settings")
groq_api = st.sidebar.text_input("Enter your Groq Api")
groq_options = st.sidebar.selectbox("Select the Groq Model",options=["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
if not groq_api:
    st.warning("Please enter the Groq API Key")
    st.stop()
else:
    llm = ChatGroq(api_key=groq_api, model=groq_options)

# Inputs
url = st.text_input("Paste the URL of the website you want to learn:") # https://lilianweng.github.io/posts/2023-06-23-agent/
user_question = st.chat_input("Ask a question:") # What is the concept of agent loop in autonomous agents?

if st.button("Learn Document") and url:
    with st.spinner("Learning Document. Please Wait..."):
        doc = WebBaseLoader(url).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        split_docs = splitter.split_documents(doc)
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        st.session_state.retriever = vectorstore.as_retriever()
        st.success("📚 Document learned successfully! Now you can write your question")


class RAGState(BaseModel):
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""


def retrieve_docs(state: RAGState) -> RAGState:
    docs = st.session_state.retriever.invoke(state.question)
    return RAGState(question=state.question,retrieved_docs=docs)

def generate_answer(state: RAGState) -> RAGState:
    context = "\n\n".join([d.page_content for d in state.retrieved_docs])
    prompt = (
        f"Answer the question based ONLY on the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state.question}")
    response = llm.invoke(prompt)
    return RAGState(
        question=state.question,
        retrieved_docs=state.retrieved_docs,
        answer=response.content)


builder = StateGraph(RAGState)
builder.add_node("retriever", retrieve_docs)
builder.add_node("responder", generate_answer)

builder.add_edge(START, "retriever")
builder.add_edge("retriever", "responder")
builder.add_edge("responder", END)

graph = builder.compile()


if user_question:
    if "retriever" not in st.session_state:
        st.warning("Please click 'Learn Document' first to load the content.")
    else:
        final_state = graph.invoke({"question": user_question})
        st.success(final_state["answer"])
