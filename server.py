import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import streamlit as st
import asyncio
from openai import AsyncOpenAI
from supabase import Client
from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps
from dotenv import load_dotenv
import subprocess
import multiprocessing

load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Pydantic AI Documentation Agent",
             description="API for querying Pydantic AI documentation. Used by both humans via Streamlit and AI agents via API.")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

class Query(BaseModel):
    question: str
    context: str = ""  # Optional context for AI agents

class Response(BaseModel):
    response: str
    source_documents: list = []  # References to source documentation

@app.get("/")
async def root():
    return {"message": "Welcome to the Pydantic AI Documentation Agent", 
            "endpoints": {
                "api": "/api/ask - POST endpoint for AI agents",
                "ui": "/ui - Streamlit interface for humans"
            }}

@app.post("/api/ask")
async def ask_question(query: Query):
    """
    API endpoint for AI agents to query Pydantic AI documentation.
    Accepts both the question and optional context about the coding task.
    """
    try:
        deps = PydanticAIDeps(
            supabase=supabase,
            openai_client=openai_client
        )
        
        # Include context in the query if provided
        full_query = f"{query.context}\n\nQuestion: {query.question}" if query.context else query.question
        
        result = await pydantic_ai_expert.run(
            full_query,
            deps=deps,
        )
        
        response_text = ""
        for message in result.messages:
            if hasattr(message, 'parts'):
                for part in message.parts:
                    if part.part_kind == 'text':
                        response_text += part.content
        
        return Response(
            response=response_text,
            source_documents=[]  # TODO: Add relevant source documents
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_streamlit():
    """Run the Streamlit server"""
    subprocess.run(["streamlit", "run", "streamlit_ui.py", 
                   "--server.port=8501", 
                   "--server.address=0.0.0.0"])

if __name__ == "__main__":
    # Start Streamlit in a separate process
    streamlit_process = multiprocessing.Process(target=run_streamlit)
    streamlit_process.start()
    
    # Start FastAPI
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
