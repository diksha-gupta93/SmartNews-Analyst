import os
import streamlit as st
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading the data....")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Splitting the text....")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    main_placeholder.text("Embedding Vector Started Building....")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    # save the FAISS vectorstore pkl file
    vectorstore_openai.save_local("faiss_store")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        vectorstore_load = FAISS.load_local(file_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore_load.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result["answer"])

        # display source if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            # split the sources with newline
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
