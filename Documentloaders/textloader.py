from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
prompt=PromptTemplate(
    template="write the summary of the {poem}",
    input_variables=['poem']
)
parser=StrOutputParser()
llm=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct")
model=ChatHuggingFace(llm=llm)
loader=TextLoader("Documentloaders/cricket.txt",encoding='utf-8')
docs=loader.load()
chain=prompt|model|parser
r=chain.invoke({'poem':docs[0].page_content})
print(r)