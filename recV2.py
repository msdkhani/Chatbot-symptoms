# recommend_physician.py
import os
import streamlit as st
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class PhysicianRecommender:
    def __init__(self):
        

        GOOGLE_API_KEY = 'AIzaSyCfIo2JXlBEQ32XRGP_f_XMSEziNpad2b8'
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

       

        self.st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        self.msgs = StreamlitChatMessageHistory()
        self.memory = ConversationBufferMemory(chat_memory=self.msgs, return_messages=True, memory_key="chat_history")

        prompt_template = '''
        Role and Purpose

        Your role is to generate a fake name and Number as a physician. When user asks for a physician you need to recommend a specialist corresponding to their symptoms but with a fake name and number.

        Conversation Flow

        Do not mention any notes such as: "this is just a sample response, I am an AI and cannot provide real medical advice or personal information like phone numbers. It's important to seek medical attention and advice from qualified healthcare professionals for any health concerns or before making any decisions related to your health or treatment" or similiar notes.Do not mention that the name is fake or any sentences like: Please note: This is just a sample response, I am an AI and cannot provide real medical advice or personal information like phone numbers.
        You should add : "Remember, this is just a recommendation, and it's always a good idea to consult with your primary care physician as well." at the end of your message. 
        '''

        avatars = {
            "human": 'https://www.growcropsonline.com/assets/img/agent-2.jpg',
            "ai": 'https://img.freepik.com/premium-photo/female-nurse-with-stethoscope-cap-3d-rendering_1057-19809.jpg'
        }

        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(prompt_template),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=True,
            memory=self.memory
        )

    def reco(self):
        response = self.conversation.predict(human_input="Please recommend me a physician with according to my symptoms and zipcode")
        # print('history: ', history)
        print('response: ', response)
        return response
