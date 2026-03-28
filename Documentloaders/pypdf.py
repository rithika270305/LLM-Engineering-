from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
llm=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct")
model=ChatHuggingFace(llm=llm)
loader=PyPDFLoader("Documentloader\dl-curriculum.pdf")
docs=loader.load()
print(docs[0].page_content)