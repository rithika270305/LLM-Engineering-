# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain.output_parsers.structured import StructuredOutputParser, ResponseSchema
# load_dotenv()
# llm=HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct')
# model=ChatHuggingFace(llm=llm)

# schema=[
#     ResponseSchema(name='fact_1',description="write fact no 1 about topic"),
#     ResponseSchema(name='fact_2',description="write fact no 2 about topic"),
#     ResponseSchema(name='fact_3',description="write fact no 3 about topic")
# ]

# parser=StructuredOutputParser.from_response_schemas(schema)
# template=PromptTemplate(
#     template="give 3 facts about the {topic} \n {inst_f}",
#     input_variables=['topic'],
#     partial_variables=[{'inst_f':parser.get_format_instruction()}]

# )

# chain=template | model |parser
# result=chain.invoke({'topic':'blackhole'})
# print(result)

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

class person(BaseModel):
    name:str=Field(description="Name of the person")
    age:int=Field(gt=18,description="Age of the person")
    city:str=Field(description="city of the person")
load_dotenv()
llm=HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct')
model=ChatHuggingFace(llm=llm)
parser=PydanticOutputParser(pydantic_object=person)

template=PromptTemplate(
    template="Generate the name, age and city of the fictional {place} person \n {inst_f}",
    input_variables=['place'],
    partial_variables={'inst_f':parser.get_format_instructions()}
)

# prompt=template.invoke({'place':'chinese'})
# result=model.invoke(prompt)
# final=parser.parse(result.content)
# print(final)

chain=template | model | parser
result=chain.invoke({'place':'chinese'})
print(result)