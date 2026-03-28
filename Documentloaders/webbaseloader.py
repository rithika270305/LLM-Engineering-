from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
url="https://www.amazon.in/Apple-MacBook-Laptop-18%E2%80%91core-40%E2%80%91core/dp/B0GR1B1ZS8/ref=sr_1_1_sspa?crid=1Q0DSI71B1L7E&dib=eyJ2IjoiMSJ9.VLeYQd5GfampurUGChUp72yrVp9O55ilHy30IcFPhGHyim8jTlNJ0u-McVVIA5ZzJd3BgdpFsA45483v2ESqVoue2MqybSFu1WfjvoyNg3eSrjM5Kc3-d3n3rr9RJ9Bcag9z7v3gh3Hb8iDNSoSwAqI4wVYHAEEm0M-qZsyFAFlPD0yu92EWQoMy3h2c47Oc3ZtQB6Q_W5o_2R13P-Z7XJQALYPRLuFyJRJzukBHkjM.50tg1ir52VMIae5wAFo9Dk7eoA-k8Ei_Zfg6g_GDuLM&dib_tag=se&keywords=macbook&qid=1773472809&sprefix=macbook%2Caps%2C446&sr=8-1-spons&aref=rT1y5lu6Th&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1"

loader=WebBaseLoader(url) # can also give list of url to load 4 5 docs at once
docs=loader.load()
llm=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct")
model=ChatHuggingFace(llm=llm)
prompt=PromptTemplate(
    template="answer the following {question} from {text}",
    input_variables=['question','text']
)
parser=StrOutputParser()
chain=prompt|model|parser
r=chain.invoke({'question':"what is the product we r taling about",'text':docs[0].page_content})
print(r)