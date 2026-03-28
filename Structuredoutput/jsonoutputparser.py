from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
llm=HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct')
model=ChatHuggingFace(llm=llm)
parser=JsonOutputParser()
template1=PromptTemplate(
    template="write 5 facts about the {topic} \n {instruction_format}",
    input_variables=['topic'],
    partial_variables={'instruction_format':parser.get_format_instructions()}
)

chain=template1 | model | parser
result=chain.invoke({'topic':'blackhole'})
print(result)