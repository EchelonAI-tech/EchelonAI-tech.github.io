import os
import streamlit as st
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
from streamlit_chat import message
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    
           
def main():
    init()

    st.set_page_config(
        page_title="ULTIMATE ECHELON GPT"
    )
    st.header("ULTIMATE ECHELON GPT")
    col1, col2, col3 = st.columns(3)
    chat = ChatOpenAI()
    prompt_1 = st.sidebar.selectbox('Primer Prompt "DEFINE AI SYSTEM ROLE:"', ["None","Author", "Instructor", "Expert", "Start-up",  ])
    model_selector = st.sidebar.selectbox('Model selection:', ["gpt-4","gpt-3.5-turbo", "DALL-E", "Leonardo", ])
    output_choice = st.sidebar.selectbox('Output selection:', ["Text","Markdown", "HTML", "List", "Email", "PDF", 'WordDoc', "PPT", "Outline", "Story", "Analysis"])
    project_directory = st.sidebar.text_input('Output file path:')
    fc_per = st.sidebar.checkbox('Fact Check')
    pot_per = st.sidebar.checkbox('Probability of Truth')
    con_per = st.sidebar.checkbox('Concise Reasoning')
    ver_per = st.sidebar.checkbox('Verbose Reasoning')
    tce_per = st.sidebar.checkbox('Tracing')
    user_input = st.text_input("ASSIGNMENT FOR AI: ", key="user_input")
    uploaded_files = st.file_uploader("Upload files", type=["pdf", "txt", "doc", "docx", "jpg"], accept_multiple_files=True)
    


    if fc_per:
        prompt_2 = "Please fault check all output and site all sources" 
    else:
        prompt_2 = ""

    if pot_per:
        prompt_3 = "Before providing your response please evaluate it for truthfulness and append and append the percentabe to your response   " 
    else:
        prompt_3 = ""

    if con_per:
        prompt_4 = "Before providing your response please evaluate and briefly summarize your thought process  " 
    else:
        prompt_4 = ""

    if ver_per:
        prompt_5 = "Before providing your response please evaluate and verbosely explain your thought process " 
    else:
        prompt_5 = ""

    if tce_per:
        prompt_6 = "Before providing your response please evaluate it in a quick summarry explain each step that was taken for the action and return  " 
    else:
        prompt_6 = ""




    # Check if user_input has been assigned a value
    if user_input:
        # Initialize message history with user_input value
        st.session_state.messages = [
            SystemMessage(content="You are a helpful " + prompt_1 + " that is an expert in " + user_input)
        ]
    else:
        # Initialize message history without user_input value
        st.session_state.messages = [
            SystemMessage(content="You are a helpful " + prompt_1)
        ]
    
       # initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful "+ prompt_1 + " that is an expert in "+ user_input)
      ]


# handle user input
    if user_input:
        st.session_state.messages.append(HumanMessage(content="provide your answer in the form of a "+output_choice+"with rich text markdown using bold text and bulletpoints where appropriate. "+ prompt_2 + prompt_3 + prompt_4 + prompt_5 + prompt_6))
    with st.spinner("Thinking..."):
        response = chat(st.session_state.messages)
        st.session_state.messages.append(
            AIMessage(content=response.content))

                # display message history
        
        
        large_text_output = st.text_area("", value=response.content, height=520, max_chars=None, key=None)
    
 




if __name__ == '__main__':
    main()
