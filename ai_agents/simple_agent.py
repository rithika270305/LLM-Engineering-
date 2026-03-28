from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
import os
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
import langchainhub 
load_dotenv()
API_KEY="5012e3c415327208cefcdd2fbc1aedcf"
@tool
def weather(city:str)->str:
    """This function returns the current weather report of any city provided"""
    url="https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API key}"
    response=requests.get(url)
    return response.json()

search_tool=DuckDuckGoSearchRun()
llm=ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct",
    api_key=os.getenv("NVIDIA_API_KEY")
)

agent=create_agent(  # created a react agent (reasoning + action)
    model=llm,
    tools=[search_tool,weather]
)
response=agent.invoke({"messages": [
        {"role": "user", "content": "What is the Capital of AndhraPradesh and what is its current temperature and humidity"}
    ]})
print(response)
