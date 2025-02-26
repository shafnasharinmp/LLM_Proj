
# Commented out IPython magic to ensure Python compatibility.- [requirements.txt]
# %pip install google-generativeai
# %pip install python-dotenv
# %pip install langchain
# %pip install PyPDF2
# %pip install chromadb
# %pip install faiss-cpu
# %pip install -qU "langchain-chroma>=0.1.2"
# %pip install -U langchain-community
# %pip install langchain_google_genai


from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv , find_dotenv

import os
import google.generativeai as genai
print(dir(genai))

load_dotenv("/content/API_Keys.env")
#print(os.getenv("GEMINI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))



# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# uploaded_files = ["/content/SJS Transcript Call.pdf"]  
# raw_text = ''
# for filename in uploaded_files:
#     with open(filename, "rb") as f:
#         raw_text += get_pdf_text([f])
# raw_text




# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
# text_chunks = get_text_chunks(raw_text)



#Function to create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key= os.getenv("GEMINI_API_KEY"))
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
# get_vector_store(text_chunks)


# Function sets up a conversational AI chain to extract structured insights
def get_conversational_chain():
    prompt_template = """
    You are an expert financial analyst evaluating a company’s earnings call transcript to extract key insights for an investor. Your goal is to provide a structured summary focused on investment potential. Analyze the transcript and extract the following insights:

    1. Future Growth Prospects
    2. Key Business Changes
    3. Growth Triggers & Investment Catalysts
    4. Risks & Material Factors Affecting Next Year’s Earnings

    Output Format: Present insights in bullet points with concise explanations and relevant financial figures. Ensure clarity and investor relevance by avoiding excessive details while maintaining accuracy.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3,google_api_key= os.getenv("GEMINI_API_KEY"))
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Function to handle user input and fetch answer
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key= os.getenv("GEMINI_API_KEY"))

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

# user_question = "Give insights of the transcript: "
# response = user_input(user_question)
# print(response)



'''Function Call'''
#----------------------------------------------------------------------------
uploaded_files = ["/content/SJS Transcript Call.pdf"]  
raw_text = ''
for filename in uploaded_files:
    with open(filename, "rb") as f:
        raw_text += get_pdf_text([f])
text_chunks = get_text_chunks(raw_text)
get_vector_store(text_chunks)

user_question = "Give insights of the transcript: "
response = user_input(user_question)


print("\n📌 **Extracted Insights:**\n")  
print("════════════════════════════════════════════════════════════════════════")  
print(response)  
print("════════════════════════════════════════════════════════════════════════")
#----------------------------------------------------------------------------

