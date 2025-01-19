from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import asyncio
from openai import AsyncOpenAI
from supabase import Client
from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Initialize clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    try:
        # Prepare dependencies
        deps = PydanticAIDeps(
            supabase=supabase,
            openai_client=openai_client
        )

        # Run the agent
        result = await pydantic_ai_expert.run(
            query.question,
            deps=deps,
        )

        # Get the response text
        response_text = ""
        for message in result.messages:
            if hasattr(message, 'parts'):
                for part in message.parts:
                    if part.part_kind == 'text':
                        response_text += part.content

        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
