from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about bumrah'
doc_embd=embedding.embed_documents(documents)
q_embd=embedding.embed_query(query)
n=np.array(cosine_similarity([q_embd],doc_embd))
index=np.argmax(n)
print(documents[index])
print("similarity score",n[0][index])

