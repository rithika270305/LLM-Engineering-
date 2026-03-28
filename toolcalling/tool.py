from langchain_core.messages import AIMessage,HumanMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import requests
from langchain_core.tools import InjectedToolArg
from langchain.tools import tool
from typing import Annotated
import os
import json
from dotenv import load_dotenv
load_dotenv()
@tool
def conversion_factor(base_currency:str,target_currency:str)->float:
    """ This function fetches the current currenxy conversion factor between given base currency
    and target currency"""

    url = f"https://v6.exchangerate-api.com/v6/85782fc38fa82fa8dc8890c6/pair/{base_currency}/{target_currency}"
    response=requests.get(url)
    return response.json()

@tool
def conversion(base_currency:int,conversion_rate:Annotated[float,InjectedToolArg])->float:
    """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """
    return base_currency * conversion_rate
print(conversion_factor.invoke({'base_currency':'USD','target_currency':'INR'}))
model = ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct",
    api_key=os.getenv("NVIDIA_API_KEY")
)
model_with_tools=model.bind_tools([conversion_factor,conversion])
messages=[HumanMessage("What is the conversion factor between INR and USD, and based on that can you convert 10 inr to usd")]
ai_message=model_with_tools.invoke(messages)
messages.append(ai_message)
print(ai_message.tool_calls)
for toolscall in ai_message.tool_calls:
    if toolscall['name']=='conversion_factor':
        toolmessage1=conversion_factor.invoke(toolscall)
        print(toolmessage1)
        conversion_rate=json.loads(toolmessage1.content)['conversion_rate']
        messages.append(toolmessage1)
    if toolscall['name']=='conversion':
        toolscall['args']['conversion_rate']=conversion_rate
        toolmessage2=conversion.invoke(toolscall)
        messages.append(toolmessage2)

result=model_with_tools.invoke(messages)
print(result)

