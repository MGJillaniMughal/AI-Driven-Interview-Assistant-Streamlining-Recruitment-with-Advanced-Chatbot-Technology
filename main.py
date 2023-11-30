import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from pydantic import BaseModel, Field, conlist
from typing import Optional
from langchain.chains import LLMChain, create_tagging_chain_pydantic
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
# Set environment variable for OpenAI API key
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"


class PersonalDetails(BaseModel):
    full_name: Optional[str] = Field(None, description="Full name of the user.")
    school_background: Optional[conlist(int, max_items=3)] = Field(
        None, description="Education background as a list of three integers representing degree level, major relevance, and college ranking.", min_items=3
    )
    working_experience: Optional[conlist(int, max_items=3)] = Field(
        None, description="Career background as a list of three integers representing job level, position relevance, and company ranking.", min_items=3
    )
    interview_motivation: Optional[int] = Field(
        None, description="Motivation level to join the interview, from 1 (not interested) to 10 (very interested)."
    )
# Initialize LLM and tagging chain
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
tagging_chain = create_tagging_chain_pydantic(PersonalDetails, llm)

# Function to generate questions based on fields that need information
def ask_for_info(ask_for):
    prompt = ChatPromptTemplate.from_template(
        """You are a job recruter who only ask questions.
        What you asking for are all and should only be in the list of "ask_for" list. 
        After you pickup a item in "ask for" list, you should extend it with 20 more words in your questions with more thoughts and guide.
        You should only ask one question at a time even if you don't get all according to the ask_for list. 
        Don't ask as a list!
        Wait for user's answers after each question. Don't make up answers.
        If the ask_for list is empty then thank them and ask how you can help them.
        Don't greet or say hi.
        ### ask_for list: {ask_for}

        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(ask_for=ask_for)

# Function to check which fields are empty in the user's details
def check_what_is_empty(user_details):
    return [field for field, value in user_details.dict().items() if value in [None, "", 0]]

# Function to update user details with non-empty fields from new data
def add_non_empty_details(current_details, new_details):
    non_empty_details = {k: v for k, v in new_details.dict().items() if v not in [None, "", 0]}
    return current_details.copy(update=non_empty_details)

# Function to filter the response and update user details
def filter_response(text_input, user_details):
    res = tagging_chain.run(text_input)
    updated_details = add_non_empty_details(user_details, res)
    ask_for = check_what_is_empty(updated_details)
    return updated_details, ask_for

# Function to create a radar chart for visualizing user details
def radar_chart(motivation, education, career):
    df = pd.DataFrame({
        "r": [motivation] + education + career,
        "theta": ['Motivation', 'Highest Degree', 'Academic Major', 'College Ranking', 'Job Level', 'Job Position', 'Company Ranking']
    })
    fig = px.line_polar(df, r='r', theta='theta', line_close=True,
                        color_discrete_sequence=px.colors.sequential.Plasma_r,
                        template="plotly_dark", title="Candidate's Job Match", range_r=[0, 10])
    st.sidebar.header('For Recruiter Only:')
    st.sidebar.write(fig)

# Streamlit UI elements
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": ask_for_info(ask_init)}]
if "details" not in st.session_state:
    st.session_state.details = PersonalDetails()
if "ask_for" not in st.session_state:
    st.session_state.ask_for = ['full_name', 'school_background', 'working_experience', 'interview_motivation']

# Displaying the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handling user input
if answer := st.chat_input("Please answer the question. "):
    st.session_state.messages.append({"role": "user", "content": answer})
    st.session_state.details, st.session_state.ask_for = filter_response(answer, st.session_state.details)
    next_question = ask_for_info(st.session_state.ask_for) if st.session_state.ask_for else \
        """Thank you for participating in this interview...
        """
    st.session_state.messages.append({"role": "assistant", "content": next_question})
    if not st.session_state.ask_for:
        final_details = st.session_state.details.dict()
        radar_chart(final_details['interview_motivation'], final_details['school_background'], final_details['working_experience'])
