from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()
model=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]
vectorstoress=FAISS.from_documents(
    documents=docs,
    embedding=model,
    

)
retriever=vectorstoress.as_retriever(
    search_strategy="mmr",
    search_kwargs={"k":3,"lambda_mult":0.6}

)
query="what is langchain?"
result=retriever.invoke(query)
print(result)