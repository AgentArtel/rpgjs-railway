from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel
from typing import List, Optional
import openai
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Supabase
supabase: Client = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_SERVICE_KEY", "")
)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    query: str

def verify_token(req: Request):
    token = req.headers.get('Authorization')
    if not token or token.split(' ')[1] != os.getenv("API_BEARER_TOKEN"):
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

@app.post("/api/ask")
async def ask_question(question: Question, token: str = Depends(verify_token)):
    try:
        # Get embedding for the question
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=question.query.replace("\n", " ")
        )
        embedding = response.data[0].embedding

        # Search for similar content in Supabase using rpgjs_pages
        matches = supabase.rpc(
            'match_rpgjs_pages',  # Updated function name
            {
                'query_embedding': embedding,
                'match_count': 5,
                'filter': {'source': 'rpgjs_docs'}
            }
        ).execute()

        if not matches.data:
            return {"answer": "I couldn't find any relevant information about that in the RPGJS documentation."}

        # Prepare context from matches
        context = "\n\n".join([
            f"Title: {match['title']}\nContent: {match['content']}"
            for match in matches.data
        ])

        # Generate response using GPT-4
        messages = [
            {"role": "system", "content": "You are an expert on RPGJS, a framework for creating RPG games in JavaScript. Answer questions based on the provided documentation context. If you're not sure about something, say so rather than making assumptions."},
            {"role": "user", "content": f"Context from RPGJS documentation:\n\n{context}\n\nQuestion: {question.query}\n\nPlease provide a clear and accurate answer based on the RPGJS documentation context provided above."}
        ]

        chat_response = openai.ChatCompletion.create(
            model=os.getenv("LLM_MODEL", "gpt-4"),
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        return {
            "answer": chat_response.choices[0].message.content,
            "sources": [
                {
                    "title": match["title"],
                    "url": match["url"],
                    "similarity": match["similarity"]
                }
                for match in matches.data
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
