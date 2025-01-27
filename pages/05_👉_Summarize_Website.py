import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from typing import List, Tuple, Dict, Any, Optional
import ollama
from utils import render_sidebar

render_sidebar()


def extract_model_names(
    models_info: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, ...]:
 
    model_names = tuple(model["model"] for model in models_info["models"])
    return model_names


def main() -> None:
    
    st.subheader("Summarize Any Webpage 💻")
    urls = st.text_input("Paste your URL..")
    
    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    # Model selection
    if available_models:
        selected_model = st.selectbox(
            "Pick a model available locally on your system ⬇️", 
             available_models,
             key="model_select"
        )
    
    message_container = st.container(height=400, border=True)
    llm = Ollama(model=selected_model)
    chain = load_summarize_chain(llm, chain_type="stuff")
    
    
    if urls:
        loader = WebBaseLoader(urls)
        docs = loader.load()
        result = chain.run(docs)  #chain.invoke
        message_container.write(result)
    else:
        st.warning("Paste URL first..")
    
    
if __name__ == "__main__":
    main()


