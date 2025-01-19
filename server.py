import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.wsgi import WSGIMiddleware
from pydantic import BaseModel
import streamlit as st
from streamlit.web.server import Server
import asyncio
from openai import AsyncOpenAI
from supabase import Client
from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

class Query(BaseModel):
    question: str

@app.post("/api/ask")
async def ask_question(query: Query):
    try:
        deps = PydanticAIDeps(
            supabase=supabase,
            openai_client=openai_client
        )
        
        result = await pydantic_ai_expert.run(
            query.question,
            deps=deps,
        )
        
        response_text = ""
        for message in result.messages:
            if hasattr(message, 'parts'):
                for part in message.parts:
                    if part.part_kind == 'text':
                        response_text += part.content
        
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount Streamlit
app.mount("/", WSGIMiddleware(Server(streamlit_ui.__name__).app))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
