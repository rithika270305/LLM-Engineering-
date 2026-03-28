from langchain_classic.retrievers import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEndpoint,HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
load_dotenv()
model=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]
vector=FAISS.from_documents(
    documents=all_docs,
    embedding=model
)
llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1"
)
multiq=MultiQueryRetriever.from_llm(
    llm=llm,
    retriever=vector.as_retriever(search_kwargs={"k":2})
)
query="how to improve energy levels and stay balanced?"
result=multiq.invoke(query)
for i,doc in enumerate(result):
    print(f"Result {i+1}")
    print("\nContent:",doc.page_content)