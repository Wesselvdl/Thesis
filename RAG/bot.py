import streamlit as st
from utils import write_message
from agent import generate_response

import time
import pandas as pd
import io
from contextlib import redirect_stdout
import re

# tag::setup[]
# Page Config
st.set_page_config("Legal JRC RAG", page_icon=":books:")
# end::setup[]

# tag::session[]
# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the Legal Retrieval Bot!  How can I help you?"},
    ]
# end::session[]

# tag::submit[]
# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        response = generate_response(message)
    # response = generate_response(message)
    write_message('assistant', response)
# end::submit[]

# tag::chat[]
# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("What is said about LFA?"):
    # Display user message in chat message container
    write_message('user', prompt)

    # Generate a response
    handle_submit(prompt)
# end::chat[]

# prompt = "Generate an action plan for the regulations with IDs 3.2.8, 4.2.1, and 5.1.1."

#     # Display user message in chat message container
# write_message('user', prompt)
# handle_submit(prompt)