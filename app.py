# RAG Emotions Chatbot

# Ran locally
# pip3 install -q -U datasets

# Import Chainlit
import chainlit as cl

# Get secret keys from .env file
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Download emotions dataset from Hugging Face Hub
from datasets import load_dataset
dataset = load_dataset("dair-ai/emotion")

# Store IMDB Dataset locally as CSV file
import pandas as pd

train_df = dataset['train'].to_pandas()
train_df.to_csv('emotions.csv', index=False)

dataset_dict = {}
dataset_dict["train"] = train_df

print(dataset_dict["train"].head())

# Install LangChain locally
# pip3 install -q -U langchain
# pip3 install -U langchain-community

from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path="emotions.csv")
data = loader.load()

len(data) # ensure we have actually loaded data into a format LangChain can recognize

# Chunk loaded data to improve retrieval performance
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # TODO: How do we create a text splitter with 1000 character chunks and 100 character overlap?
    chunk_size=1000,
    chunk_overlap=100
)

#Convert data into a single string
text = "\n".join(str(doc) for doc in data)

#Chunk the data
chunked_documents = text_splitter.split_text(text)

len(chunked_documents) # ensure we have actually split the data into chunks

# Use OpenAI embeddings to create a vector store, locally
# pip3 install -q -U langchain-openai

#import os
#from google.colab import userdata
from langchain_openai import OpenAIEmbeddings

openai_api_key = OPENAI_API_KEY
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
# TODO: How do we create our embedding model?

# Create embedder
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

store = LocalFileStore("/path/to/root") # TODO: How do we create a local file store to for our cached embeddings?
embedder = OpenAIEmbeddings(openai_api_key=openai_api_key) # TODO: How do we create our embedder?

# Create vector store using FAISS, locally
# pip3 install -q faiss-cpu tiktoken

from langchain_community.vectorstores import FAISS


vector_store = FAISS.from_texts(texts=chunked_documents, embedding=embedder) # TODO: How do we create our vector store using FAISS?

# LocalFileStore instance was created previously, store

vector_store.save_local("faiss_index")

# Ask RAG system a question
query = "I did a lot of cool things today like meet a friend for lunch and go horseback riding"

query = str(query)

# Embed query
embedded_query = embedder.embed_query(query)

# Similarity search to find documents similar to our query
similar_documents = vector_store.similarity_search_by_vector(embedded_query)

# Print similar documents that the similarity search returns
for page in similar_documents:
  print(page.page_content)

# Create components
# pip install -q langchain_openai, done earlier

from langchain_core.runnables.base import RunnableSequence
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Create the components (chefs)

# Create prompt template to send to our LLM that will incorporate from our retriever with the question we ask chat model
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly and empathetic coach bot who helps humans understand their emotions better."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks! Tell me a bit about your day?"),
        ("human", "{question}"),
    ]
)

# Create retriever for our documents
retriever = vector_store.as_retriever()

# Create a chat model/LLM
chat_model = ChatOpenAI(api_key=openai_api_key)

# Create a parser to parse output of LLM
parser = StrOutputParser()

# Chain output of retriever, prompt, model, and parser to get a good answer to our query
runnable_chain = RunnableSequence(
      prompt_template,
      chat_model,
      parser
)

# Synchronous execution
input_data = {"question": query}

output_chunks = runnable_chain.invoke(input_data)


# Chainlit execution
@cl.on_message
async def main(message: cl.Message):
    query = message.content
    input_data = {"question": query}
    response = runnable_chain.invoke(input_data)
    await cl.Message(response).send()
