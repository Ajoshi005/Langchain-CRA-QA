# Import dependencies
import streamlit as st
import os
import pinecone
import openai
from tqdm.autonotebook import tqdm
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
import tiktoken
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']
#st.secrets["OPENAI_API_KEY"]

pinecone.api_key = os.environ['PINECONE_API_KEY']
#st.secrets["PINECONE_API_KEY"]

#basic chatcompletion API
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)









#----------BUILDING the APP.py----------------------------#

# Set Streamlit app title with emojis of a parrot and a chain
st.title("🦜 Function Calling with OpenAI  📄")

# Define the colors for the Government of Canada color palette
palette = {
    "primary": "#c11919",
    "secondary": "#157e2c",
    "background": "#ceb9b9",
    "text": "#3a3d3a",
    "base": "light",
    "font": "monospace",
}

# Set page background color and text color
st.markdown(
    f"""
    <style>
    body {{
        background-color: {palette["background"]};
        color: {palette["text"]};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Get user input from text area
user_input = st.text_area("Please enter your query on personal tax filing for the year 2022-23:")
# Check if the user has entered any input
if user_input:
    # Process user input and get the model's output
    model_output = response["choices"][0]["message"]["content"]
    tokens_used = response["usage"]["total_tokens"]
    # Display the output in formatted text format
    st.markdown(
        f"""
        <div style='background-color: {palette["secondary"]}; padding: 10px;'>
            <p style='color: {palette["primary"]}; font-size: 18px;'>
                {model_output}
            </p>
            <p style='color: {palette["primary"]}; font-size: 24px;'>
                {"Tokens Used:"}
            </p>
            <p style='color: {palette["primary"]}; font-size: 18px;'>
                {tokens_used}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

#-----------------SIDEBAR--------------

# Add a heading to the sidebar
st.sidebar.header("About the App")

# Add a description to the sidebar
st.sidebar.write(
       "Description HERE"
)
sidebar_text = """
Tech stack:
1) LangChain - for LLM App implementation
2) Open AI - For vector embedding and LLM for creating responses
3) Pinecone - Creating and storing CRA text data in vector embedding for similarity search(cosine)
4) Streamlit - For App UI and hosting
"""

st.sidebar.markdown(sidebar_text)

# Add a link to your resume on LinkedIn in the sidebar
linkedin_url = "https://www.linkedin.com/in/akashjoshi/"
st.sidebar.markdown('<a href="'+linkedin_url+'"><img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width=30 height=30>', unsafe_allow_html=True)
# GitHub Logo and Link
github_url = "https://github.com/Ajoshi005/Langchain-CRA-QA"
st.sidebar.markdown('<a href="'+github_url+'"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width=30 height=30></a>', unsafe_allow_html=True)

