from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()
llm=HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct",
)
model=ChatHuggingFace(llm=llm)
st.header("Research Paper Summarizer")

paper_input=st.selectbox("Select Research Paper Name",["Attention is all you need","BERT:pre-training of deep bidirectional transformer","GPT-4:LANGUAGE MODELS ARE FEW-SHOT LEARNERS","Diffusion models beats GANs on Image Synthesis"])
style_input=st.selectbox("Select Explanation Style",["Begineer-Friendly","Technical","Code-oriented","Mathematical"])
length_input=st.selectbox("Select output length explanation",["Short paragraph","Medium paragraph","Long paragraph"])

template=load_prompt("template.json")  # now we made it reusable
# filling the place holders 
prompt=template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input

})
if st.button("Summarize"):
    result=model.invoke(prompt)
    st.write(result.content)

