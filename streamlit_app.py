from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
import streamlit as st
import os
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


nvapi_key = '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'
os.environ["NVIDIA_API_KEY"] = nvapi_key


st.set_page_config(page_title="Symptom extraction", page_icon="")
st.title("Health Chatbot: Symptoms extraction")


prompt_template = ''''
Greeting :
Agent: How are you doing today? Do you have any health concerns or symptoms you would like to discuss?

Agent:
Your primary role is to gather detailed information about users’ medical symptoms. Please ask relevant, clear, and concise questions to capture the patient’s symptoms accurately. 
Do not diagnose or provide treatment. Ask for more details about the symptoms.



Always respond as “Agent:” 

Template:

	1.	Primary Symptom:
	•	What is the main symptom or issue you are experiencing?
	2.	Severity:
	•	On a scale of 1 to 10, how severe is this symptom?
	•	Does the severity fluctuate throughout the day?
	3.	Duration:
	•	How long have you been experiencing this symptom?
	•	Did it start suddenly or gradually?
	4.	Pain:
	•	Are you experiencing any pain? If so, where is the pain located?
	•	Can you describe the pain (e.g., sharp, dull, throbbing)?
	•	How long does the pain last when it occurs?
 	5.	Additional Symptoms:
 	•	Are you experiencing any other symptoms? (e.g., fever, fatigue, nausea, dizziness)
 	•	If yes, please list and describe them.
 	6.	Previous Episodes:
 	•	Have you experienced this symptom before?
 	•	If yes, when and how often?
 	7.	Triggers and Relievers:
 	•	Have you noticed any factors that trigger or worsen your symptoms?
 	•	Are there any activities or treatments that relieve your symptoms?
 	8.	Medical History:
 	•	Do you have any existing medical conditions or a history of similar issues?
 	•	Are you currently taking any medications or supplements?
 	9.	Lifestyle Factors:
 	•	Have there been any recent changes in your lifestyle, such as diet, exercise, or stress levels?
 	10.	Additional Information:
 	•	Is there anything else you think might be relevant to your symptoms?
Extract at the end Summary:

	•	Symptoms:
	•	Symptom 1: [Name] - Duration: [Time] - Severity: [Scale]
	•	Symptom 2: [Name] - Duration: [Time] - Severity: [Scale]
	•	…
	•	Notes:
	•	[Additional relevant information]
 
 END conversation with the following message:
 We will send your information to a healthcare professional for further evaluation. Please seek medical attention if your symptoms worsen or if you experience any emergency symptoms. Thank you for sharing your information with us.
'''


msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

if prompt := st.chat_input(placeholder="How Do you feel today?"):
    st.chat_message("user").write(prompt)

    llm = ChatNVIDIA(model="ai-llama3-70b", nvidia_api_key=nvapi_key, max_tokens=200)
    # temprature of llm
    llm.temperature = 0.02
    # top_p of llm
    llm.top_p = 0.9


    # Combine prompt template with user input
    complete_prompt = f"{prompt_template}\n\nUser: {prompt}\nAgent:"

    # Set up conversational chat agent and executor
    tools = []
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        return_messages=True,
    )

# Display a loading spinner or message while generating response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            #st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            cfg = RunnableConfig()
            #cfg["callbacks"] = [st_cb]
            response = executor.invoke(complete_prompt, cfg)
        st.write(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]