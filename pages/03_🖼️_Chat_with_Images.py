from io import BytesIO
from IPython.display import HTML, display
import requests
import json
import base64
import streamlit as st
from PIL import Image
import tempfile
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import ollama
from typing import List, Tuple, Dict, Any, Optional, Generator
from utils import render_sidebar

render_sidebar()




def extract_model_names(
    models_info: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, ...]:
 
    model_names = tuple(model["model"] for model in models_info["models"])
    return model_names

def prompt_func(data):
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]


def main():
    st.header("CHATBOT For Images 🖼️",divider="gray")  
    st.write("Ask any question related to uploaded images 🧑‍💻")   
    
    # Get available models
    available_models = extract_model_names(ollama.list())

    # Model selection
    if available_models:
        selected_model = st.selectbox(
            "Pick a model available locally on your system ⬇️", 
            available_models,
            key="model_select"
        )
        
    llm = ChatOllama(model=selected_model, temperature=0)
    
    def results(image_b64):
        chain = prompt_func | llm | StrOutputParser()
    
        query_chain = chain.invoke(
            {"text": "解决图片中给出的问题并提供问题的答案", "image": image_b64}
        )
        return query_chain    
    
    upload_image = st.file_uploader(
        "Upload your Images here and click on 'Process' ⬇️" )#, accept_multiple_files=True)
        
    if upload_image is not None:
        st.write("Image Uploaded Sucessfully")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(upload_image.read())
            temp_file_path = temp_file.name
    
          
    process_button = st.button("Process")
    
    message_container = st.container(height=400, border=True)
    
    if (process_button and upload_image is not None):
        with st.spinner("Processing"):                
            result = results(temp_file_path)
            message_container.write(result)
    else:
        if (process_button and upload_image is None):
            st.warning("Please upload an Image first..")
            
    
    user_question = st.chat_input("Ask a question about your image")    
    
    
if __name__ == '__main__':
    main()  




































