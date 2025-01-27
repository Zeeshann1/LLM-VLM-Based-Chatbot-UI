import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from typing import List, Tuple, Dict, Any, Optional
import ollama
from utils import render_sidebar

render_sidebar()


def extract_model_names(
    models_info: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, ...]:
 
    model_names = tuple(model["model"] for model in models_info["models"])
    return model_names


def create_vector_db(urls) -> Chroma:
   
    loader = WebBaseLoader(urls)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Updated embeddings configuration
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="myRAG"
    )

    #shutil.rmtree(temp_dir)
    return vector_db

def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    
    # Initialize LLM
    llm = ChatOllama(model=selected_model)
    
    # Query prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 2
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    # Set up retriever
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    # RAG prompt template
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Create chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    return response



def main() -> None:
    
    st.subheader("Chatbot with Any Website 💻")
    urls = st.text_input("Paste your URL..")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    
    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    # Model selection
    if available_models:
        selected_model = st.selectbox(
            "Pick a model available locally on your system ⬇️", 
             available_models,
             key="model_select"
        )
        message_container = st.container(height=500, border=True)
        if urls:
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing uploaded PDF..."):
                    st.session_state["vector_db"] = create_vector_db(urls)
                    
    
    # Display chat history
    for i, message in enumerate(st.session_state["messages"]):
        avatar = "🤖" if message["role"] == "assistant" else "👤"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Chat input and processing
    if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
        try:
            # Add user message to chat
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with message_container.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            # Process and display assistant response
            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner(":green[processing...]"):
                    if st.session_state["vector_db"] is not None:
                        response = process_question(
                            prompt, st.session_state["vector_db"], selected_model
                        )
                        st.markdown(response)
                    else:
                        if st.session_state["vector_db"] is None:
                            st.warning("Please paste a URL first..")
                            
            # Add assistant response to chat history
            if st.session_state["vector_db"] is not None:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response}
                )
               
        except Exception as e:
            st.error(e, icon="⛔️")
    else:
        if st.session_state["vector_db"] is None:
            st.warning("Please paste a URL to start chat..")
    
    
if __name__ == "__main__":
    main()


