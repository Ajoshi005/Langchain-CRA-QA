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
# get openaai and pinecone api key from secrets file
openai.api_key = st.secrets["OPENAI_API_KEY"]
#os.environ['OPENAI_API_KEY']
pinecone.api_key = st.secrets["PINECONE_API_KEY"]
#os.environ['PINECONE_API_KEY']
# HF_API_KEY=os.environ['HF_API_KEY']
# get openai api key from platform.openai.com
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'

# Load pdf file
# Can also use directory loader for multiple text files. For this 1 doc I will use basic TextLoader
# Txt file path = "/Users/akashjoshi/Desktop/Python_Learning/streamlit_folder/CRAdoc.txt"

loader = TextLoader("/Users/akashjoshi/Desktop/Python_Learning/streamlit_folder/CRAdoc.txt")
CRAdoc = loader.load()

# SPLIT TEXTS
tokenizer = tiktoken.get_encoding('p50k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, #default 1000.chunk size is dependent on how coherent or random sentences are in a para
    chunk_overlap=100,#default 200.usually 10-20% to mantain the sentence struct
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)
# split data here
#Signature: text_splitter.split_text(text: 'str') -> 'List[str]'
#Docstring: Split incoming text and return chunks.
chunks = text_splitter.split_text(CRAdoc[0].page_content)

#CREATE EMBEDDINGS
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
#os.getenv("OPENAI_API_KEY")
model_name = 'text-embedding-ada-002' #Embedding model with N-Dim = 1536 ,
# i.e each vector is represented in 1536 dim space

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

#PINECONE VECTOR DB
#pip install "pinecone-client[grpc]"
index_name = 'langchain-retrieval-augmentation'

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
#os.getenv("PINECONE_API_KEY")

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment='us-west1-gcp-free'
)

if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )
#Already Run below statement once
#crasearch = Pinecone.from_texts(texts=chunks, embedding=embed, index_name=index_name)
text_field = "text"

index1 = pinecone.Index(index_name)

vectorstore = Pinecone(
    index1, embed.embed_query, text_field
)

#CREATE PROMPT TEMPLATE
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,
 just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Explain like a 10year old can understand. 
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}


# LLMQACHAIN
# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0,
    verbose=True
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    chain_type_kwargs=chain_type_kwargs,
    retriever=vectorstore.as_retriever()
)

#INPUT TAKEN FROM APP.PY
# Define the function predict_tax_query with corrected indentation
def predict_tax_query(user_input):
    with get_openai_callback() as cb:
        response = qa.run(user_input)
    return response

#----------BUILDING the APP.py----------------------------#

# Set Streamlit app title with emojis of a parrot and a chain
st.title("🦜 Your Personal Tax Assistant  📄")

# Define the colors for the Government of Canada color palette
govt_canada_palette = {
    "primary": "#25265e",
    "secondary": "#157e2c",
    "background": "#c3c3cb",
    "text": "#080808",
}

# Set page background color and text color
st.markdown(
    f"""
    <style>
    body {{
        background-color: {govt_canada_palette["background"]};
        color: {govt_canada_palette["text"]};
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
    model_output = predict_tax_query(user_input)

    # Display the output in formatted text format
    st.markdown(
        f"""
        <div style='background-color: {govt_canada_palette["secondary"]}; padding: 10px;'>
            <p style='color: {govt_canada_palette["primary"]}; font-size: 18px;'>
                {model_output}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Add the disclaimer at the bottom of the page
st.markdown(
    """
    #### Disclaimer:
    The information provided by this application is for general informational purposes only and should not be considered
     as tax advice. All answers to the queries are generated by an AI model and may not be entirely accurate or 
     up-to-date. It is essential to consult a certified tax professional from the Canada Revenue Agency (CRA) or 
     other qualified experts for personalized and accurate tax-related advice.
    """,
    unsafe_allow_html=True,
)
