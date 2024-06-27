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
Greet once:
User Name = Emily
user: -hi, hello, how are you
Agent: Hi, User Name! how are you doing today? How can I assist you with your health concerns?
 
 
 
You can find your and the user's previous messages in {history}.
You need to follow the Questions using the {input}.
 
Continue the conversation based on the Questions.
If user ask question: Answer the questions to the best of your ability.
 
If the user is not feeling well or has symptoms in {input}:
- Acknowledge their symptom once with sympathy, and then proceed with relevant questions without repeating the sympathy for each response.
 
Please Keep the questions in the same order as the template. Do not ask the same question twice. Do not ask more questions.
 
Do not repeat the history in the conversation.
Try to continue the conversation based on the Questions.
If the user mentions any symptoms, do not ask for the same symptom again.
Do not write phrases like "Here's my response based on the template."
Do not repeat what users mentioned in the conversation.
 
Questions Flow:
 
We need to collect the following information from you:
 
1.Additional Symptoms:
• Are you experiencing any other symptoms?
• If yes, please list and describe them.
 
2.Severity:
• On a scale of 1 to 10, how severe is this symptom?
 
3.Duration:
• How long have you been experiencing this symptom?
 
4.Physiscian
. Do you consider your condition as an emergency?

8. END conversation with the following message:
Please seek medical attention if your symptoms worsen or if you experience any emergency symptoms. Thank you for sharing your information with us.
 
'''

    input_variables = ['input', 'history']
    prompt_1 = PromptTemplate(template=prompt_template, input_variables=input_variables)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                 temperature=0.1, top_p=0.7)
    
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
