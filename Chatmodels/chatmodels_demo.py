from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
chatm=ChatOpenAI(model="gpt-4")
r=chatm.invoke("What is the capital of AMERICA?")
print(r) # extra meta data as well
print(r.content) # to fetch only the content i.e., answer

