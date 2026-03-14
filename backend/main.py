import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import uuid
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Data Analyst API",
    description="AI-powered data analysis using Groq + LangChain",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_datasets: dict = {}
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class QueryRequest(BaseModel):
    session_id: str
    question: str
    selected_files: Optional[list[str]] = None


class QueryResponse(BaseModel):
    answer: str
    chart: Optional[str] = None
    query_type: str
    sql_query: Optional[str] = None


class InsightRequest(BaseModel):
    session_id: str
    selected_files: Optional[list[str]] = None


@app.get("/")
def root():
    return {"message": "AI Data Analyst API is running 🚀", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    session_id: Optional[str] = None
):
    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in uploaded_datasets:
        uploaded_datasets[session_id] = {}

    uploaded_names = []

    for file in files:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a CSV file")

        contents = await file.read()
        file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"

        with open(file_path, "wb") as f:
            f.write(contents)

        df = pd.read_csv(file_path)
        uploaded_datasets[session_id][file.filename] = df
        uploaded_names.append(file.filename)
        logger.info(f"Uploaded: {file.filename} | Shape: {df.shape}")

    return {
        "session_id": session_id,
        "uploaded_files": uploaded_names,
        "message": f"Successfully uploaded {len(uploaded_names)} file(s)"
    }


@app.get("/files/{session_id}")
def list_files(session_id: str):
    if session_id not in uploaded_datasets:
        raise HTTPException(status_code=404, detail="Session not found")

    files_info = {}
    for filename, df in uploaded_datasets[session_id].items():
        files_info[filename] = {
            "rows": len(df),
            "columns": list(df.columns),
            "shape": list(df.shape)
        }
    return {"session_id": session_id, "files": files_info}


@app.get("/preview/{session_id}/{filename}")
def preview_file(session_id: str, filename: str, rows: int = 5):
    if session_id not in uploaded_datasets:
        raise HTTPException(status_code=404, detail="Session not found")
    if filename not in uploaded_datasets[session_id]:
        raise HTTPException(status_code=404, detail="File not found")

    df = uploaded_datasets[session_id][filename]
    return {
        "filename": filename,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "preview": df.head(rows).to_dict(orient="records")
    }


@app.post("/query", response_model=QueryResponse)
async def query_data(request: QueryRequest):
    session_id = request.session_id
    if session_id not in uploaded_datasets:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a CSV first.")

    datasets = uploaded_datasets[session_id]
    if request.selected_files:
        datasets = {k: v for k, v in datasets.items() if k in request.selected_files}
    if not datasets:
        raise HTTPException(status_code=400, detail="No datasets available.")

    try:
        from llm_agent import DataAnalystAgent
        agent = DataAnalystAgent()
        result = agent.run(question=request.question, datasets=datasets)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.post("/insights")
async def get_insights(request: InsightRequest):
    session_id = request.session_id
    if session_id not in uploaded_datasets:
        raise HTTPException(status_code=404, detail="Session not found.")

    datasets = uploaded_datasets[session_id]
    if request.selected_files:
        datasets = {k: v for k, v in datasets.items() if k in request.selected_files}

    try:
        from llm_agent import DataAnalystAgent
        agent = DataAnalystAgent()
        insights = agent.generate_insights(datasets=datasets)
        return {"insights": insights}
    except Exception as e:
        logger.error(f"Insights error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Insights error: {str(e)}")


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id in uploaded_datasets:
        del uploaded_datasets[session_id]
        return {"message": "Session deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")