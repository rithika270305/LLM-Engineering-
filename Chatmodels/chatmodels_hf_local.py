from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()
llm=HuggingFacePipeline.from_model_id(
    model_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.5,
                         max_new_tokens=100)

)
model=ChatHuggingFace(llm=llm)
result=model.invoke("What is the capital of India?")
print(result )