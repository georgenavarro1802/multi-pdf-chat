import time

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from html_templates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store_using_openai(text_chunks):
    # this is using OpenAI (it involves payment, but is cheap)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


def get_vector_store_using_instructor(text_chunks):
    # this is using instructor-xl (free)
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


def get_conversation_chain(vector_store):
    # using OpenAI llm
    llm = ChatOpenAI()
    # # using HuggingFace model llm
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    # # Example to print it out the response in the ui (json object with the entire history)
    # st.write(response)
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Procesing"):
                # # # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # # example of how to write the entire pdfs texts
                # st.write(raw_text)

                # # # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # # example of how to write the chunks
                # st.write(text_chunks)

                # # # create vector store (tested using both OpenAI and Instructor-xl from huggingface)
                start = time.perf_counter()
                vector_store = get_vector_store_using_openai(text_chunks)
                end = time.perf_counter()
                st.write(f"VectorStore (using OpenAI, done in {(end - start):.2f}s)")

                # start = time.perf_counter()
                # vector_store = get_vector_store_using_instructor(text_chunks)
                # end = time.perf_counter()
                # st.write(f"VectorStore (using instructor-xl {(end - start):.2f}s)")

                # # # create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)


if __name__ == '__main__':
    main()


