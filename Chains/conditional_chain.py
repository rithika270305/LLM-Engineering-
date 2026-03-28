from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import  PromptTemplate
from langchain_core.runnables import RunnableBranch,RunnableLambda
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal
load_dotenv()
llm1=HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-R1')
model1=ChatHuggingFace(llm=llm1)
llm2=HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-R1')
model2=ChatHuggingFace(llm=llm2)
parser=StrOutputParser()
class review(BaseModel):
    sentiment:Literal['positive','negative']
parser2=PydanticOutputParser(pydantic_object=review)
prompt1=PromptTemplate(
    template='classify the sentiment of the feedback {feedback}',
    input_variables=['feedback']
)
classifier_part=prompt1|model1|parser2
prompt2=PromptTemplate(
    template="give an appropriate response for the positive feedback {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instaruction':parser2.get_format_instructions()}
)
prompt3=PromptTemplate(
    template="give an appropriate response for the negative feedback {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instaruction':parser2.get_format_instructions()}
)
branch_part=RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2|model2|parser),
    (lambda x:x.sentiment=='negative',prompt3|model2|parser),
    RunnableLambda(lambda x:"could not find the sentiment ")

)

final_chain=classifier_part|branch_part
result=final_chain.invoke(
    {'feedback':"the experience with this product is the epitome example of how a smartphone should be"}
)
print(result)
final_chain.get_graph().print_ascii()