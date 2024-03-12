import pandas as pd

import os

from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

import streamlit as st
import time

import configparser

#from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler


st.set_page_config(
        page_title="LaFonte SmartBot",
        page_icon="https://www.lafonte.ch/hubfs/Favicon.png",
    )

user_avatar = "http://learn.lugano.ch/wp-content/uploads/2023/12/user.svg"
bot_avatar = "http://learn.lugano.ch/wp-content/uploads/2023/12/bot.svg"

class SimpleCallback(BaseCallbackHandler):

    def __init__(self):
        pass

    def on_llm_start(self, serialized, prompts, **kwargs):
        print("start")
        self.full_response = ""


    def on_llm_end(self, *args, **kwargs) -> None:
        st.session_state.message_placeholder.markdown(self.full_response, unsafe_allow_html=True)

    
    def on_llm_new_token(self, token: str, **kwargs):
        """Run on new LLM token. Only available when streaming is enabled."""
        self.full_response += token
        st.session_state.message_placeholder.markdown(self.full_response, unsafe_allow_html=True)
        
st.markdown(
    """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Albert+Sans:ital,wght@0,400;1,100&display=swap');
            span, p, textarea, div {
                font-family: 'Albert Sans', sans-serif;
            }
            i.e1se5lgy2 {
                border-color: #DA291C rgba(49, 51, 63, 0.2) rgba(49, 51, 63, 0.2);
            }

            .st-emotion-cache-1kyxreq.e115fcil2 {
                position: sticky;
                top: 0;
                background-color: white; /* Add any other styles you want here */
                z-index: 1000; /* Adjust the z-index as needed */
            }

            .stButton {
                color: grey;
            }

            div.inner-tab, .inner-tab li {
                color: #4F4F4F;
                font-size: 14px;
            }

            .chat-headline {
                text-align: center;
                margin-bottom: 10px;
            }

            .st-emotion-cache-eczf16 {
                display: none;
            }
        </style>
    """,
    unsafe_allow_html=True
)

from pprint import pprint

config = configparser.ConfigParser()
config.read('config.ini')

OPENAI_API_KEY = config["open-ai"]["open-ai-key"]

def load_dataset(dataset_name:str="dataset.csv"):
    """
    Funzione helper per caricare il dataset

    Args:
        dataset_name (str, optional): Nome del file salvato dalla fase di estrazione. Defaults to "dataset.csv".

    Returns:
        pd.DataFrame: DataFrame Pandas dei dati raccolti da LangChain
    """
    data_dir = "./data"
    file_path = os.path.join(data_dir, dataset_name)
    df = pd.read_csv(file_path)
    df["url"] = "https://shop.lafonte.ch/products/" + df["Handle"]
    df = df[df["Published"] == True]
    df = df.dropna(subset=["SEO Description"])
    return df

def create_chunks(dataset:pd.DataFrame, chunk_size:int, chunk_overlap:int):
    """
    Crea chunk informazionali dal dataset 

    Args:
        dataset (pd.DataFrame): Dataset Pandas
        chunk_size (int): Quanti chunk informazionali?
        chunk_overlap (int): Quanti chunk condivisi?

    Returns:
        list: lista di chunk
    """
    text_chunks = DataFrameLoader(
        dataset, page_content_column="SEO Description"
    ).load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, length_function=len
        )
    )
    # aggiungiamo i metadati ai chunk stessi per facilitare il lavoro di recupero
    for doc in text_chunks:
        title = doc.metadata["Title"]
        type = doc.metadata["Type"]
        content = doc.page_content
        url = doc.metadata["url"]
        image = doc.metadata["Image Src"]
        final_content = f"TITLE: {title}\TYPE: {type}\BODY: {content}\nURL: {url}\nIMAGE: {image}"
        doc.page_content = final_content

    return text_chunks

def create_or_get_vector_store(chunks: list) -> FAISS:
    """
    Funzione per creare o caricare il database vettoriale dalla memoria locale

    Returns:
        FAISS: Vector store
    """
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # embeddings = HuggingFaceInstructEmbeddings() # 

    if not os.path.exists("./db"):
        print("CREATING DB")
        vectorstore = FAISS.from_documents(
            chunks, embeddings
        )
        vectorstore.save_local("./db")
    else:
        print("LOADING DB")
        vectorstore = FAISS.load_local("./db", embeddings)

    return vectorstore

def get_conversation_chain(vector_store: FAISS, system_message:str, human_message:str) -> ConversationalRetrievalChain:
    """
    Oggetto LangChain che permette domanda-risposta tra umano e LLM

    Args:
        vector_store (FAISS): Vector store
        system_message (str): System message
        human_message (str): Human message

    Returns:
        ConversationalRetrievalChain: Chatbot conversation chain
    """

    callback = SimpleCallback()

    llm = ChatOpenAI(model="gpt-4", 
                     streaming=True, 
                     openai_api_key=OPENAI_API_KEY,
                     callbacks=[callback]) # possiamo cambiare modello a piacimento
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages(
                [
                    system_message,
                    human_message,
                ]
            ),
        },
    )
    return conversation_chain

def handle_style_and_responses(user_question: str, conversation_chain: ConversationalRetrievalChain) -> None:

    with st.chat_message("user", avatar=user_avatar):
        st.markdown(user_question)

    with st.chat_message("assistant", avatar=bot_avatar):
        st.session_state.message_placeholder = st.empty()
        response = conversation_chain({"question": user_question})

    st.session_state.chat_history = response["chat_history"]

def main():

    # Load or create necessary resources
    df = load_dataset("lafonte_products.csv")
    chunks = create_chunks(df, 1000, 0)

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        """
        You are a chatbot tasked with helping the user find the right product to buy.

        You should read carefully the description of each product you know and suggest the best fit with user needs.
        Do not repeat products that you already suggested and if you do not find products that fit with user needs do not recommend random product, but say that there is no item with such features.

        If you are asked a question that is not about products or finding products, reply by saying that the question is out of scope.
        
        For each product recommendation, provide also the image.

        Given a question, you should respond with the most relevant documentation page by following the relevant context below:\n
        {context}

        If you need to provide a link, use the following HTML format:
        <a href="URL" target="_blank">Text to display</a>

        If you need to provide an image, use the following HTML format:
        </br><img src="IMAGE" width="250" height="250"></br>

        If you need to provide a list, always use HTML formatting. For example:
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
        """
    )
    
    human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = create_or_get_vector_store(chunks)

    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(st.session_state.vector_store, system_message_prompt, human_message_prompt)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    
    st.image("images/lafonte-logo.svg", output_format="SVG", width=180)
    st.title("LaFonte SmartBot", anchor=False)

    st.markdown(
        """
        Il regalo perfetto per te e per i tuoi cari, a portata di domanda!
        """
    )

    #st.divider()

    st.markdown("<h3 class='chat-headline'>Come posso aiutarti?</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)


    
    # Display messages in the Streamlit app
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            role = "user"
            avatar_url = user_avatar
            name = "Tu"
        else:
            role = "assistant"
            avatar_url = bot_avatar
            name = "SmartBot"

        with st.chat_message(role, avatar=avatar_url):
            st.markdown(f"<span><b>{name}</b></span><p style='text-align: left;'>{message.content}</p>", unsafe_allow_html=True)

    user_question = st.chat_input("Cosa vuoi chiedere?")

   
    if user_question:
        handle_style_and_responses(user_question, st.session_state.conversation)





if __name__ == "__main__":
    main()

