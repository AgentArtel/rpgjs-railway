from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are an expert at RPGJS - a powerful game development framework for creating web-based RPG games.
You have access to all the RPGJS documentation, including guides, API references, and examples.

IMPORTANT:
1. Only provide RPGJS-specific solutions, not generic JavaScript game development patterns
2. Always reference specific RPGJS classes, methods, and features from the documentation
3. Include version information when relevant
4. If a solution requires generic JavaScript, clearly distinguish between RPGJS-specific and generic code

Your responses should focus on RPGJS's unique features:
- RPGJS-specific event system
- Built-in RPG mechanics
- RPGJS plugin architecture
- RPGJS map and tileset handling

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering
unless you have already seen the relevant documentation.

When you first look at the documentation, always start with RAG.
Then check the list of available documentation pages and retrieve specific page content if it will help.

Always be honest when you can't find an answer in the documentation or if a URL doesn't exist.
Keep your responses friendly and conversational, like you're pair programming with the developer.
"""

rpgjs_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@rpgjs_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_rpgjs_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            # Truncate content if it's too long
            content = doc['content']
            if len(content) > 300:
                content = content[:300] + "..."
                
            chunk_text = f"""
# {doc['title']}

{content}

🔗 Source: [{doc['url']}]({doc['url']}) (Similarity: {doc['similarity']:.2f})
"""
            formatted_chunks.append(chunk_text)
            
        # Add a summary of sources at the end
        sources_summary = "\n\n---\n\n### 📚 Documentation Sources:\n"
        for i, doc in enumerate(result.data, 1):
            sources_summary += f"{i}. [{doc['url']}]({doc['url']}) (Similarity: {doc['similarity']:.2f})\n"
            
        # Join all chunks with a separator and add sources summary
        return "\n\n---\n\n".join(formatted_chunks) + sources_summary
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@rpgjs_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available RPGJS documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs
        result = ctx.deps.supabase.from_('rpgjs_pages') \
            .select('url') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@rpgjs_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('rpgjs_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title']
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"