import os
import streamlit as st
import asyncio

# Create a new event loop
loop = asyncio.new_event_loop()

# Set the new event loop as the default one for the current context
asyncio.set_event_loop(loop)
from nemoguardrails import LLMRails, RailsConfig
from langchain.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain.chains import LLMChain
#prompt_templete import
from langchain_core.prompts import PromptTemplate



st.set_page_config(page_title="Symptom extraction", page_icon="")

st.title("Health Chatbot: Symptoms extraction")

nvapi_key = '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'

# Set up the NVIDIA API key: ask the user to provide it in the side bar
#st.sidebar.markdown("Please provide your NVIDIA API key:")
#nvapi_key = st.sidebar.text_input("NVIDIA API key")
os.environ["NVIDIA_API_KEY"] = nvapi_key




msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
 

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I assist you today with your health concerns?")  # Custom initial message

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        st.write(msg.content)
        
if prompt := st.chat_input(placeholder="How do you feel today?"):
    st.chat_message("user").write(prompt)
    # if not nvapi_key:
    #     st.error("Please provide your NVIDIA API key in the sidebar.")
    #     st.stop()
    prompt_template = ''''
    Greeting:
    user: Hi, I am not feeling well today., I have some symptoms I would like to discuss., Hi, How are you
    Agent: How are you doing today? Do you have any health concerns or symptoms you would like to discuss?
    Have sympothy if they mention any symptoms or concerns.

    Agent:
    Your primary role is to gather detailed information about users’ medical symptoms. Please ask relevant, clear, and concise questions to capture the patient’s symptoms accurately. Do not diagnose or provide treatment advice. Ask for more details about the symptoms.
    If the user Mention any symptoms relevent to the template, do not ask for the same symptom again.

    Always respond as “Agent:” 

    Template:

    We need to collect the following information from you:

    Primary Symptom:
        • Can you describe the main symptom or issue you are experiencing?

    Severity:
        • On a scale of 1 to 10, how severe is this symptom?
        • Does the severity fluctuate throughout the day?

    Duration:
        • How long have you been experiencing this symptom?
        • Did it start suddenly or gradually?

    Pain:
        • Are you experiencing any pain? If so, where is the pain located?
        • Can you describe the pain (e.g., sharp, dull, throbbing)?
        • How long does the pain last when it occurs?

    Additional Symptoms:
        • Are you experiencing any other symptoms? (e.g., fever, fatigue, nausea, dizziness)
        • If yes, please list and describe them.

    Previous Episodes:
        • Have you experienced this symptom before?
        • If yes, when and how often?

    Triggers and Relievers:
        • Have you noticed any factors that trigger or worsen your symptoms?
        • Are there any activities or treatments that relieve your symptoms?

    Medical History:
        • Do you have any existing medical conditions or a history of similar issues?
        • Are you currently taking any medications or supplements?

    Lifestyle Factors:
        • Have there been any recent changes in your lifestyle, such as diet, exercise, or stress levels?

    Additional Information:
        • Is there anything else you think might be relevant to your symptoms?

    Extract at the end Summary:
        • Symptoms:
            • Symptom 1: [Name] - Duration: [Time] - Severity: [Scale]
            • Symptom 2: [Name] - Duration: [Time] - Severity: [Scale]
            • …
        • Notes:
            • [Additional relevant information]

    END conversation with the following message:
    We will send your information to a healthcare professional for further evaluation. Please seek medical attention if your symptoms worsen or if you experience any emergency symptoms. Thank you for sharing your information with us.'''


    config = RailsConfig.from_path("./config")
    rails = LLMRails(config=config)
    input_variables = ["topic"]
    prompt_1 = PromptTemplate(template=prompt_template, input_variables=input_variables)
    
    chain = LLMChain(
        llm=rails.llm,
        prompt=prompt_1,
        verbose=True,
        memory=memory,
    )
    
    rails.register_action(chain, name="bot")
    st.write(memory.buffer)
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            cfg = RunnableConfig()
            cfg["callbacks"] = [st_cb]
            st.write(["callbacks"])
            
            response = rails.generate(prompt=prompt)
        st.write(response)
