import requests
import json
import base64
import streamlit as st
from PIL import Image
import tempfile
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from utils import render_sidebar

render_sidebar()


def chat_with_image(image_path):
    
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llava",
        "prompt": "What is in this picture",
        "stream": False,
        "images": [image_base64]
    }

    response = requests.post(url, data=json.dumps(payload))
    return response


def main():

    st.header("Lava MOdel Chatbot for Images \U0001F916",divider="gray")  
    st.write("Ask any question related to uploaded images \U0001F9D1\u200D\U0001F4BB")   

    
    #with col2:

    user_question = st.chat_input("Ask a question about your image")
    

    #with col1:

    upload_image = st.file_uploader(
        "Upload your Images here and click on 'Process" )#, accept_multiple_files=True)
        
    message_container = st.container(height=500, border=True)
          
        #This part is to view Image
    if upload_image is not None:
        #image = Image.open(upload_image)
        #st.image(image, caption= "Uploaded Image.", use_column_width=True)
        st.write("Image Uploaded Sucessfully")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(upload_image.read())
            temp_file_path = temp_file.name
            
        
    if st.button("Process"):
        with st.spinner("Processing"):
            result = chat_with_image(temp_file_path)
            #with col2.container
            message_container.write(result.text)
                    
 
if __name__ == '__main__':
    main()    
    
    
    
    
    
    
    
    
    
    
    
    
    
