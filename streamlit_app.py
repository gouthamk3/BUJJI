from click import prompt
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from pypdf import PdfReader
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import base64
from dotenv import dotenv_values
import os

# Initialize API key
OPENAI_API_KEY = dotenv_values(".env").get("OPENAI_API_KEY")

st.set_page_config(page_title="KONU")

# Encode the image
with open("Konu logo.png", "rb") as img_file:
    encoded_img = base64.b64encode(img_file.read()).decode()

# Add image in markdown
st.markdown(
    f"""
    <div>
        <img src="data:image/png;base64,{encoded_img}" alt="KONU Logo" width="200">
    </div>
    <div style="text-align: center;">   
        <h1 style="margin-top: 10px;">KONU BUJJI</h1>
    </div>
    """,
    unsafe_allow_html=True
)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.header("KONU chatbot")
    with st.form("question_form"):
        user_question = st.text_input("### Ask a Question", key="user_question")
        submitted = st.form_submit_button("Submit")
    # Process the user's question if the form is submitted
    if submitted and user_question:
        user_input(user_question) 

#To Upload New Document remove # 
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button"):  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
               
if __name__ == "__main__":
    main()
