from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
text="Hi I AM RITHIKA!"
mod=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
result=mod.embed_query(text)
print(str(result))