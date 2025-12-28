import os
from typing import Dict, Any

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI



load_dotenv()


PIPELINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "pipeline_A": {
        "chunk_size": 300,
        "chunk_overlap": 50,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "pipeline_B": {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    },
    "pipeline_C": {
        "chunk_size": 800,
        "chunk_overlap": 150,
        "embedding_model": "sentence-transformers/paraphrase-MiniLM-L6-v2",
    },
    "pipeline_D": {
        "chunk_size": 1200,
        "chunk_overlap": 200,
        "embedding_model": "sentence-transformers/paraphrase-mpnet-base-v2",
    },
}

VECTOR_DBS: Dict[str, FAISS] = {}

PIPELINE_META: Dict[str, Dict[str, Any]] = {}


def load_and_store(path: str) -> None:
   
    global VECTOR_DBS, PIPELINE_META

    
    loader = PyPDFLoader(path)
    docs = loader.load()

    VECTOR_DBS = {}
    PIPELINE_META = {}

    
    for name, cfg in PIPELINE_CONFIGS.items():
        print(f"[{name}] Building index with {cfg}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name=cfg["embedding_model"]
        )

        db = FAISS.from_documents(chunks, embeddings)

        VECTOR_DBS[name] = db
        PIPELINE_META[name] = {
            "chunk_size": cfg["chunk_size"],
            "chunk_overlap": cfg["chunk_overlap"],
            "embedding_model": cfg["embedding_model"],
            "num_chunks": len(chunks),
        }


def ask_question(q: str, k: int = 4) -> Dict[str, Any]:
 
    if not VECTOR_DBS:
        raise ValueError("No pipelines are built yet. Upload a PDF first.")

    openai_key = os.getenv("OPENAI_API_KEY")
    use_llm = bool(openai_key)

    llm = None
    if use_llm:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)   

    pipeline_results: Dict[str, Any] = {}

    for name, db in VECTOR_DBS.items():
        cfg = PIPELINE_META.get(name, {})

        
        retrieved_docs = db.similarity_search(q, k=k)
        
        contexts = []

        for d in retrieved_docs:
            contexts.append({
                "text":d.page_content,
                "page":d.metadata.get('page'),
                "pdf_title":os.path.basename(d.metadata.get("source", "")),
            })

       
        if not use_llm:
            pipeline_results[name] = {
                "config": cfg,
                
                "contexts": contexts,
            }
            continue

      
        prompt = f"""
        Answer the user's question
        USING ONLY the context below. If the answer is not in the context,
        say you don't know.

        PIPELINE: {name}
        CONTEXT:
        {contexts}

        QUESTION: {q}
        """

        resp = llm.invoke(prompt)

        pipeline_results[name] = {
            "config": cfg,
            "answer": resp.content,
            "contexts": contexts,
        }

    return {
        "mode": "rag_with_llm" if use_llm else "retrieval_only",
        "question": q,
        "pipelines": pipeline_results,
    }
