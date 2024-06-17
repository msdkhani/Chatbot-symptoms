
import os
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


# os.environ["NVIDIA_API_KEY"] = "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"

import re
from typing import List, Union

import requests
from bs4 import BeautifulSoup

def html_document_loader(url: Union[str, bytes]) -> str:
    """
    Loads the HTML content of a document from a given URL and return it's content.

    Args:
        url: The URL of the document.

    Returns:
        The content of the document.

    Raises:
        Exception: If there is an error while making the HTTP request.

    """
    try:
        response = requests.get(url)
        html_content = response.text
    except Exception as e:
        print(f"Failed to load {url} due to exception {e}")
        return ""

    try:
        # Create a Beautiful Soup object to parse html
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style tags
        for script in soup(["script", "style"]):
            script.extract()

        # Get the plain text from the HTML document
        text = soup.get_text()

        # Remove excess whitespace and newlines
        text = re.sub("\s+", " ", text).strip()

        return text
    except Exception as e:
        print(f"Exception {e} while loading document")
        return ""
def create_embeddings(embedding_path: str = "./embed"):
    
    embedding_path = "./embed"
    print(f"Storing embeddings to {embedding_path}")

    # List of web pages containing NVIDIA Triton technical documentation
    urls = [
         "https://www.froedtert.com/doctors?clinicaffiliations=222&sort=relevance&show_map=1&distance=25&s=1",
         "https://www.froedtert.com/doctors?clinicaffiliations=222&distance=25&s=1&show_map=1&sort=relevance&page=3"
    ]

    documents = []
    role = """ You are a medical assistant.
Your task is to help patients find out the right physician at Frodert Hospital based on their symptoms and the doctor's specialty.

User Symptoms: They are given by the User.

Physician Information: Provide information about the physicians at Frodert Hospital. Include their specialties, expertise, and name. Do not provide any other information according to the physician.

Match Symptoms to Physician: Based on the user's symptoms, match them to the most suitable physician at Frodert Hospital. Consider the physician's specialty and expertise in treating similar conditions.

Connect User to Physician: Once you have identified the right physician, provide the user with the physician's name, speciality and number.
just return the doctor name
Follow-up: After connecting the user to the physician, offer to follow up with them to ensure they received the care they needed. Below is the physician's information:
"""

    documents.append(role)

    for url in urls:
        document = html_document_loader(url)
        documents.append(document)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
    )
    texts = text_splitter.create_documents(documents)
    index_docs(url, text_splitter, texts, embedding_path)
    print("Generated embedding successfully")

def index_docs(url: Union[str, bytes], splitter, documents: List[str], dest_embed_dir) -> None:
    """
    Split the document into chunks and create embeddings for the document

    Args:
        url: Source url for the document.
        splitter: Splitter used to split the document
        documents: list of documents whose embeddings needs to be created
        dest_embed_dir: destination directory for embeddings

    Returns:
        None
    """
    embeddings = NVIDIAEmbeddings(model="nvolveqa_40k",nvidia_api_key="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")

    for document in documents:
        texts = splitter.split_text(document.page_content)

        # metadata to attach to document
        metadatas = [document.metadata]

        # create embeddings and add to vector store
        if os.path.exists(dest_embed_dir):
            update = FAISS.load_local(folder_path=dest_embed_dir, embeddings=embeddings,allow_dangerous_deserialization=True)
            # update = FAISS.load_local(folder_path=dest_embed_dir, embeddings=embeddings)
            update.add_texts(texts, metadatas=metadatas)
            update.save_local(folder_path=dest_embed_dir)
        else:
            docsearch = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
            docsearch.save_local(folder_path=dest_embed_dir)



class reco():
    def __init__(self) -> None:
        create_embeddings()

        self.embedding_model = NVIDIAEmbeddings(model="nvolveqa_40k")

        self.embedding_path = "embed/"
        self.docsearch = FAISS.load_local(folder_path=self.embedding_path, embeddings=self.embedding_model, allow_dangerous_deserialization=True)

        from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
        prompt_template = """
        Prompt:

        You are a question generator. Your task is to extract the symptoms from the {question} and create a question that starts with "Which doctor at Froedtert Hospital should I visit as I have" followed by the extracted symptoms.

        Input:

        You said you have {symptom1}. Are you experiencing any other symptoms?

        Output:

        Which doctor at Froedtert Hospital should I visit as I have {symptom1}?
"""
    
# based on the extracted symptoms use {context} to find the relevent docter. just return the doctor name based on it. 
# dont ask follow up question 


        prompt_template = """You are a medical assistant.
Your task is to help patients find out the right physician at Frodert Hospital based on their symptoms and the doctor's specialty.

User Symptoms: has to extract from the 
start info:
{question}
end info.

Physician Information: Provide information about the physicians at Frodert Hospital. Include their specialties, expertise, and name. Do not provide any other information according to the physician.

Match Symptoms to Physician: Based on the user's symptoms, match them to the most suitable physician at Frodert Hospital. Consider the physician's specialty and expertise in treating similar conditions.

Connect User to Physician: Once you have identified the right physician, provide the user with the physician's name, speciality and number.
just return the doctor name
Follow-up: After connecting the user to the physician, offer to follow up with them to ensure they received the care they needed. Below is the physician's information:
{context}
        """
        TEST_PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt_template)



        # Create prompt from prompt template
        # self.prompt = PromptTemplate(
        #     input_variables=["question"],
        #     template=prompt_template,
        # )

        self.llm = ChatNVIDIA(model="llama2_70b")

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT)

        self.chat = ChatNVIDIA(model="mixtral_8x7b",nvidia_api_key="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", temperature=0.1, max_tokens=100, top_p=1.0)
        self.doc_chain = load_qa_chain(self.chat , chain_type="stuff", prompt=TEST_PROMPT)
        self.qa = ConversationalRetrievalChain(
            retriever=self.docsearch.as_retriever(),
            combine_docs_chain=self.doc_chain,
            memory=self.memory,
            question_generator=self.question_generator,
        )
    import re





    def call_fn(self,User_text):

        query = User_text

        print('oomade inja',query)
        
        result = self.qa({"question": query})
        print(result)
        return result.get("answer")




