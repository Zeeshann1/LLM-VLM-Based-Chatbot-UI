import streamlit as st

def render_sidebar():
    st.set_page_config(layout="wide")
    st.sidebar.header("LLM & VLM Powered AI Chatbot")
    st.sidebar.image("assets/logo.jpg", use_container_width=True)
    st.sidebar.markdown("---")
