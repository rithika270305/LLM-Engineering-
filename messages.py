from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
load_dotenv()
llm=HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct')
model=ChatHuggingFace(llm=llm)
message=[
    SystemMessage(content="You are a helpful AI Assistant"),
    HumanMessage(content="Tell me about Langchain")
]
result=model.invoke(message)
message.append(AIMessage(content=result.content))
print(message)