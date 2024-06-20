# CarePilot
## NVIDIA LangChain DocsAgent

### Demo Video
Click on the image below to watch the demo video of the "CarePilot" on youtube:

[![Demo Video](https://github.com/msdkhani/Chatbot-symptoms/assets/76404542/11b23dcb-8b12-431b-9178-2f349455b28a)]([https://www.youtube.com/watch?v=Q40DGRHC0XA](https://youtu.be/uS876oiPcgU)


### Impact
Leveraging advanced large language models (LLMs), our chatbot offers a user-friendly interface to simplify healthcare navigation. A 2019 survey revealed that 63% of patients struggle with the healthcare system, facing issues like understanding treatment options (51%), finding the right doctor (49%), and comprehending medical costs (47%). Complexity and unclear roles of healthcare providers add to these challenges, particularly for marginalized communities, resulting in significant healthcare inequities and nearly $320 billion in annual spending. Additionally, the fragmentation of healthcare into over 120 specialties makes finding the right care difficult. Our chatbot addresses these issues by using LLMs to identify the relevant physician based on symptoms. It considers the user’s location and insurance details to recommend suitable doctors and facilitate appointment bookings, ensuring timely medical intervention and improved health outcomes. By bridging the gap between patients and healthcare providers, our chatbot enhances healthcare accessibility, particularly in low-resource regions, aiming to significantly improve patient care.
### Use Case
Use natural language to ask chatbot about your health issues, it would ask relevant questions to your symptoms, to extract as much information.

### Instructions
 - Clone the repository from GitHub
 - Navigate to the project directory
 - Install the required Python packages: pip install -r requirements.txt
 - Run the application using Streamlit: streamlit run chat.py

### Workflow Diagram:


### What Was Used in This Project
- **Langchain**:
 ConversationalRetrievalChain and LLMChain from langchain.chains for conversational retrieval and language model chains.
 QuestionGenerationChain from langchain.chains.question_answering for question generation.
 ConversationBufferMemory from langchain.memory for managing conversation memory.
 FAISS from langchain.vectorstores for managing embeddings and similarity search.

- **NVIDIAEmbeddings**:
 NVIDIA embeddings are used to create and manage embeddings for documents.

- **Langchain_nvidia_ai_endpoints, NVIDIA AI Foundation endpoints**:
  This package is used to integrate NVIDIA AI Foundation endpoints with LangChain. It provides access to NVIDIA's advanced AI models for embedding generation, document ranking, and large language model (LLM) interactions.
 relevant documents based on user queries.

- **NVIDIA NIM API (meta/llama3-70b)**:
  This API provides the large language model used for generating the final response to the user's query. It processes the crafted prompt and produces a coherent and accurate answer.


- **Streamlit**:
Used for building interactive web applications for data science and machine learning.

- **Nemoguardrails**:
LLMRails and RailsConfig from nemoguardrails for managing guardrails.

- **Chains**:
LLMChain from langchain.chains for language model chaining.

- **Langchain Core**:
PromptTemplate from langchain_core.prompts for defining prompt templates.
StrOutputParser from langchain_core.output_parsers for parsing model output.

