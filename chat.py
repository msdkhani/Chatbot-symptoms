import os
import streamlit as st
#import logging
import asyncio
# Create a new event loop
loop = asyncio.new_event_loop()
# Set the new event loop as the default one for the current context
asyncio.set_event_loop(loop)

from nemoguardrails import LLMRails, RailsConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA
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
        }
        .st-emotion-cache-p4micv {
            width: 3rem;
            height: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("CarePilot")

nvapi_key = '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'
os.environ["NVIDIA_API_KEY"] = nvapi_key

if "history" not in st.session_state:
    st.session_state.history = [{"role": "context", "content": ""}]

st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True, memory_key="chat_history")



prompt_template = '''
Greeting:
User Name = Emily
user: -hi, hello, how are you
Agent: Hi,Emily! how are you doing today? How can I assist you with your health concerns?


You need to follow the template using the {input}.
You can find your and the user's previous messages in {history}.

Continue the conversation based on the user's {input}.
If user ask question: Answer the questions to the best of your ability.

If the user is not feeling well or has symptoms in {input}:
- Acknowledge their symptom once with sympathy, and then proceed with relevant questions without repeating the sympathy for each response.

Please ask relevant, clear, and concise questions to capture the patient’s symptoms accurately and naturally.
Do not diagnose or provide treatment advice.
If the user mentions any symptoms relevant to the template, do not ask for the same symptom again.

Do not repeat the history in the conversation.
Try to continue the conversation based on the template.
If the user mentions any symptoms, do not ask for the same symptom again.
The user might mention multiple symptoms; ask about all of them.
Do not write phrases like "Here's my response based on the template."

Template:

We need to collect the following information from you:

1.Additional Symptoms:
• Are you experiencing any other symptoms? (e.g., fever, fatigue, nausea, dizziness)
• If yes, please list and describe them.

2.Severity:
• On a scale of 1 to 10, how severe is this symptom?
• Does the severity fluctuate throughout the day?

3.Duration:
• How long have you been experiencing this symptom?
• Did it start suddenly or gradually?


4.Additional Information:
• Is there anything else you think might be relevant to your symptoms?

5.Insurance information:
• Do you have insurance? If so, what is your insurance provider?

6.Doctor Appointment:
• Have you scheduled an appointment with a specific doctor, or would you like help finding a doctor in your area?

Extract_Summary =
• Symptoms:
    • Symptom 1: [Name] - Duration: [Time] - Severity: [Scale] - previous episodes: [detail], medical history: [detail], insurance: [provider]
    • Symptom 2: [Name] - Duration: [Time] - Severity: [Scale] - previous episodes: [detail], medical history: [detail], insurance: [provider]
    • …
• Notes:
    • [other relevant information]

END conversation with the following message:
Please seek medical attention if your symptoms worsen or if you experience any emergency symptoms. Thank you for sharing your information with us.

'''


input_variables = ['input', 'history']
prompt_1 = PromptTemplate(template=prompt_template, input_variables=input_variables)
llm = ChatNVIDIA(model="meta/llama3-70b-instruct", nvidia_api_key=nvapi_key, max_tokens=150)

config = RailsConfig.from_path("config")
guardrails = RunnableRails(config)

output_parser = StrOutputParser()


chain = prompt_1 | (guardrails | llm) | output_parser

if len(msgs.messages) == 0:
    msgs.clear()

avatars = {
    "human": 'https://static01.nyt.com/images/2018/09/21/books/00edim1/merlin_137623458_968f9cfc-ceb1-449e-b5ce-543fffca302b-articleLarge.jpg?quality=75&auto=webp&disable=upscale',
    "ai": 'https://media.istockphoto.com/id/1329569957/photo/happy-young-female-doctor-looking-at-camera.jpg?s=612x612&w=0&k=20&c=7Wq_Y2cl0T4op6Wg_3DFc-xtZfCqTTDvfaXkPGyrHDM='
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
            response = chain.invoke({'input': prompt, 'history': st.session_state.history})
            st.session_state.history.append({"role": 'assistant', "content": response})
            msgs.add_ai_message(response)

        st.write(response)