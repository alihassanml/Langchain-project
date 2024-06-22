import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import PIL.Image
import os  
import tempfile 
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

load_dotenv()


def Image_Processing(text, image_path):
    llm = ChatGoogleGenerativeAI(model='gemini-pro-vision', api_key=os.getenv('GOOGLE_API_KEY'))
    message = HumanMessage(
        content=[
            {
                'type': 'text',
                'text': text,
            },
            {'type': 'image_url', 'image_url': image_path},
        ]
    )
    result = llm.invoke([message])
    return result.content


def main():

   
    st.sidebar.caption('Streamlit web application that integrates with LangChain and Google Generative AI for processing text alongside images')
    st.sidebar.subheader('Follow Me')
    st.sidebar.link_button("Connect LinkedIn",'https://www.linkedin.com/in/alihassanml')
    st.sidebar.link_button("Connect On Github",'https://github.com/alihassanml',type="secondary")
    st.sidebar.caption('Develop by: Ali Hassan')

    st.title('LangChain Image Processing Model')
    user_input = st.text_input('Enter Text')
    image = st.file_uploader('Upload File', type=['jpg', 'png', 'jpeg'])
    submit = st.button('Ask Question')

    if submit and image is not None:  
        user_image = PIL.Image.open(image)
        st.image(user_image, caption='Image Uploaded', use_column_width=True)

    # Image Save            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            user_image.save(temp_file.name)
            image_path = temp_file.name



        response = Image_Processing(user_input, image_path) 
        st.subheader('Response')
        st.write(response)
        os.remove(image_path)


def Chat():
    st.title('LangChain Chat Model')
    user_input = st.text_input('Enter Text')
    submit = st.button('Ask Question')
    st.sidebar.caption('Streamlit web application that integrates with LangChain and Google Generative AI for processing text alongside images')
    st.sidebar.subheader('Follow Me')
    st.sidebar.link_button("Connect LinkedIn",'https://www.linkedin.com/in/alihassanml')
    st.sidebar.link_button("Connect On Github",'https://github.com/alihassanml',type="secondary")
    st.sidebar.caption('Develop by: Ali Hassan')
    if submit:
        llm = ChatGoogleGenerativeAI(model='gemini-pro', api_key=os.getenv('GOOGLE_API_KEY'))
        response = llm.invoke(user_input)
        st.subheader('Response')
        st.write(response.content)
    



st.sidebar.title('Chose Langchain Model')
pages = st.sidebar.selectbox(
   
    "Chose Langchain Model",
    ("Langchain Pdf","Langchain Chat Model", "Image Processing")
    )






def Pdf_load(uploaded_file):
    text=""
    for pdf in uploaded_file:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def TextSplitter(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def Embedding(chunks):
    embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001', api_key=os.environ['GOOGLE_API_KEY'])
    vector = FAISS.from_texts(chunks, embedding=embedding)
    vector.save_local('faiss_index')

def conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the context provided. If answer not available then answer is yes or no. Don't give wrong answer.
    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

def user_input(user_question):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001",api_key=os.environ['GOOGLE_API_KEY'])
    new_db = FAISS.load_local("faiss_index", embedding,allow_dangerous_deserialization=True)  
    docs = new_db.similarity_search(user_question)
    chain = conversation_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.header('Response')
    st.write("Reply: ", response["output_text"])




def pdf():
    st.title('LangChain Gemini Pdf Reader')
    user_input_text = st.text_input('Enter question')
    submit_question = st.button('Ask Question')
    if submit_question:
        user_input(user_input_text)
    
    with st.sidebar:
        st.sidebar.header('Upload File')
        pdf = st.sidebar.file_uploader('Choose File', type='pdf', accept_multiple_files=True)
        submit_file = st.sidebar.button('Submit File')
        if submit_file:
            response = Pdf_load(pdf)
            splitter = TextSplitter(response)
            Embedding(splitter)
            st.success("Done")




if pages == 'Image Processing':
    main()

if pages == 'Langchain Chat Model':
    Chat()

if pages == 'Langchain Pdf':
    pdf()