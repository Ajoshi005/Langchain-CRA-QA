

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