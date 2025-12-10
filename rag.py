import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI

load_dotenv()

db = None

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_and_store(path: str):
    global db
    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
    )

    chunk = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL_NAME)
    db = FAISS.from_documents(chunk, embeddings)




def ask_question(q: str, k: int = 4):
    global db
    if db is None:
        raise ValueError("Database not initialized. Please load documents first.")
    ResDocs = db.similarity_search(q, k)                         #list of doc objs r returned 
    context = [doc.page_content for doc in ResDocs]
    context_str = "\n\n--\n\n".join(context)

    openai_api_key = os.getenv("OPENAI_KEY")

    if not openai_api_key:
        return {
            'mode':'retrievel only',
            'question':q,
            'answer':context_str
        }
    
    llm = ChatOpenAI(
        model_name = "gpt-4o-mini",
        temperature=0
    )

    prompt = f""" Answer the user's question USING ONLY the context below. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context:{context_str}
    question: {q}"""

    response = llm.invoke(prompt)

    return{
        'mode': 'retrieval-augmented generation',
        'question': q,
        'answer': response.content,
        'context_used' : context_str
    }









    

