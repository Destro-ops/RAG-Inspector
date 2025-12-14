import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from app.rag import load_and_store, ask_question, VECTOR_DBS, PIPELINE_META

app = FastAPI(title="Rag quality checker backend")

app.add_middleware(
    CORSMiddleware, 
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "uploads")

os.makedirs(DATA_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message":"The backend is running"}

@app.get("/pipelines")
def get_pipelines():
    if not VECTOR_DBS:
        return "No pipelines available. please upload a pdf first."
    
    return{"pipelines": PIPELINE_META}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="need a pdf file only")
    
    file_path = os.path.join(DATA_DIR, file.filename)

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException (status_code=500, detail = f"failed to save file:{e}")
    
    try:
        load_and_store(file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error Message: {e}")
    
    return {"status": "ok", "detail": f"Indexed {file.filename}"}

@app.post("/ask")
async def ask(question: str = Form(...), K: Optional[int] = Form(4)):
    if not question or question.strip() == "":
        raise HTTPException (status_code=400, detail="the question cannot be empty")

    try:
        results = ask_question(q = question, k=K)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ask_question failed {e}")
    except ValueError as e:
        raise HTTPException(status_code=400,detail=f"{e}")

    return results

 

    

    









