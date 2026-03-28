from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from dotenv import load_dotenv
load_dotenv()
llm=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct")
model=ChatHuggingFace(llm=llm)
chat_history=[]
with open('chathistory.txt','r') as f:
    chat_history.extend(f.readlines())

#dynamic chatprompt template
template=ChatPromptTemplate(
    [
        ('system','you are a {domain} expert'),
        # MessagesPlaceholder(variable_name='chat_history'), #how message placeholder is used
        ('human','explain about {topic}')
    ]
)
prompt=template.invoke({'domain':'cricket','topic':'wicket'})
while True:  #context less chatbot 
    user_input=input("YOU:")
    chat_history.append(HumanMessage(content=user_input))
    if user_input =='exit':
        break
    result=model.invoke(chat_history) #flexible enough to work with list of commands as well
    chat_history.append(AIMessage(content=result.content))
    print("AI:",result.content)

print(chat_history)











































































