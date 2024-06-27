import os
import streamlit as st

import asyncio
# Create a new event loop
loop = asyncio.new_event_loop()
# Set the new event loop as the default one for the current context
asyncio.set_event_loop(loop)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_core.output_parsers import StrOutputParser

#logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Symptom Extraction", page_icon="")

st.markdown(
    """
    <style>
        .st-emotion-cache-janbn0 {
            flex-direction: row-reverse;
            text-align: right;
            background-color: rgb(111 176 255 / 50%);
        }
        .st-emotion-cache-p4micv {
            width: 4rem;
            height: 4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("CarePilot")

GOOGLE_API_KEY = 'AIzaSyDh8ITSK6Aiim10eyGFJQEbEqoGmsmaZ_I'

# Set up the NVIDIA API key: ask the user to provide it in the side bar
#st.sidebar.markdown("Please provide your NVIDIA API key:")
#nvapi_key = st.sidebar.text_input("NVIDIA API key")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

if "history" not in st.session_state:
    st.session_state.history = [{"role": "context", "content": ""}]

st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True, memory_key="chat_history")

if len(msgs.messages) == 0 :
    from recommend_physician import reco
    st.session_state.doctor_recom = reco()
    prompt_template = '''
You can find your and the user's previous messages in {history}.
You need to follow the Questions using the {input}.

Do not repeat the conversation. Do not repeat history. Start acting normal form the start.
 
Role and Purpose

You are CarePilot, an AI assistant designed to collect patient signs and symptoms. Your primary task is to gather three key pieces of information: symptoms, duration, and severity. After collecting everything if the user wants the specialist have them type sepcialist to find the nearest specialist.

Conversation Flow

Engage in a conversation to collect each piece of information separately to ensure a natural flow.

Emergency Protocol

If at any point it appears to be an emergency, immediately instruct the user to contact emergency services and do not continue the conversation.

Handling Specific Emergency Situations

General Instructions for Emergencies

1.    Stay Calm: Encourage the patient to stay calm.

2.    Contact Emergency Services: Immediately instruct the patient to call emergency services (911 or local emergency number).

3.    Provide Basic Guidance: Offer simple, clear instructions based on the situation.

4.    Stop Collecting Information: Do not continue collecting further information once an emergency is identified.

Summary

CarePilot should:

•    Maintain a calm and reassuring tone.

•    Instruct the patient to contact emergency services.

•    Provide basic first aid instructions relevant to the situation.

•    Stop further information collection once an emergency is identified.
 
'''

    input_variables = ['input', 'history']
    prompt_1 = PromptTemplate(template=prompt_template, input_variables=input_variables)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    output_parser = StrOutputParser()


    st.session_state.chain = prompt_1 |  llm | output_parser

    msgs.clear()

avatars = {
    "human": 'https://www.growcropsonline.com/assets/img/agent-2.jpg',
    "ai": 'https://img.freepik.com/premium-photo/female-nurse-with-stethoscope-cap-3d-rendering_1057-19809.jpg'
}


for idx, msg in enumerate(msgs.messages):
    with st.chat_message(msg.type, avatar=avatars[msg.type]):
        st.write(msg.content)

if prompt := st.chat_input(placeholder="How do you feel today?"):
    st.chat_message("user", avatar=avatars["human"]).write(prompt)
    with st.chat_message("assistant", avatar=avatars["ai"]):
        with st.spinner("Generating response..."):
            msgs.add_user_message(prompt)
            st.session_state.history.append({"role": 'user', "content": prompt})
            response = st.session_state.chain.invoke({'input': prompt, 'history': st.session_state.history})
            st.session_state.history.append({"role": 'assistant', "content": response})
            msgs.add_ai_message(response)

        if 'sepcialist' in prompt.lower() or 'sepcialist' in prompt.lower() or 'urgent' in prompt.lower():
            print(response)
            response = st.session_state.doctor_recom.call_fn(response)
            msgs.add_ai_message(response)
    st.write(response)
