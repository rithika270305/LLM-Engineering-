from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict,Annotated,List,Optional,Literal

load_dotenv()
llm=HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct')
model=ChatHuggingFace(llm=llm)

#schema

class Review(TypedDict):
    key_topics:Annotated[List[str],"write down all the key topics discussed about in the sumary in a list"]
    summary:Annotated[str,"a brief summary of the review"] # to add extra description for the work
    sentiment:Annotated[Literal["pos","neg"],"return sentiment of the review either positive,negative or neutral"]
    pros:Annotated[Optional[List[str]],"write down all the pros in a list"]
    cons:Annotated[Optional[List[str]],"write down all the cons in a list"]
    name:Annotated[Optional[str],"write the name of the reviewer only if there is a name(pronoun)"]

structured_model_output=model.with_structured_output(Review)

result=structured_model_output.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
""")

print(type(result))
print(result)
print(result['sentiment'])
print(result['name'])