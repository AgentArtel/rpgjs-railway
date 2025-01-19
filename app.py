import os
import streamlit as st
from supabase import create_client
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

def get_embedding(text: str):
    """Get embedding vector from OpenAI."""
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def search_docs(query: str, limit: int = 5):
    """Search documentation using embeddings."""
    query_embedding = get_embedding(query)
    
    # Use the match_rpgjs_pages function to find similar chunks
    response = supabase.rpc(
        'match_rpgjs_pages',
        {
            'query_embedding': query_embedding,
            'match_count': limit,
            'filter': {}
        }
    ).execute()
    
    return response.data

# Streamlit UI
st.title("RPGJS Documentation Search")
st.write("Ask any question about RPGJS!")

# Search box
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Searching..."):
        results = search_docs(query)
        
        if not results:
            st.warning("No relevant documentation found. Try rephrasing your question.")
        else:
            for idx, result in enumerate(results, 1):
                st.markdown(f"### Result {idx}")
                st.markdown(f"**Source:** [{result['url']}]({result['url']})")
                st.markdown(f"**Title:** {result['title']}")
                st.markdown(f"**Content:**\n{result['content']}")
                st.markdown(f"**Similarity Score:** {result['similarity']:.2f}")
                st.markdown("---")
