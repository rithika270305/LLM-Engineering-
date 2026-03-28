from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
model=OpenAIEmbeddings(model=" ",dimensions=64)
documents=[] #list of documents then we do model.embed_documents()
r=model.embed_query("what is your name?") # to embed a single query
print(str(r))