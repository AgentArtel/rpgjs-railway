# RPGJS Documentation Agent

An AI-powered documentation assistant for RPGJS, built with Streamlit and OpenAI. This agent helps developers find and understand RPGJS documentation through natural language queries.

## Features

- **AI-Powered Responses**: Uses GPT-4 to provide contextual, accurate answers about RPGJS
- **Smart Search**: Employs embeddings-based search to find relevant documentation
- **Conversational Interface**: Chat-based UI for natural interaction
- **Source Links**: Every response includes links to original documentation
- **Relevance Scores**: Shows similarity scores to help evaluate source relevance
- **Real-time Streaming**: Responses stream in real-time for better UX

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Database**: Supabase (PostgreSQL with pgvector)
- **AI/ML**: 
  - OpenAI GPT-4 for response generation
  - OpenAI Ada-002 for embeddings
- **Framework**: Pydantic AI for agent behavior

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/AgentArtel/rpgjs-railway.git
   cd rpgjs-railway
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file with:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   LLM_MODEL=gpt-4o-mini
   ```

4. **Initialize the database**
   ```bash
   psql -h your_host -d your_database -U your_user -f site_pages.sql
   ```

5. **Run the application**
   ```bash
   # Start the FastAPI server
   uvicorn api:app --reload
   
   # In another terminal, start the Streamlit interface
   streamlit run streamlit_ui.py
   ```

## Usage

1. Open the Streamlit interface in your browser
2. Type your question about RPGJS in the chat input
3. The agent will:
   - Search through the documentation
   - Find relevant information
   - Generate a helpful response
   - Provide links to source documentation

Example questions:
- "How do I create a new map in RPGJS?"
- "What are player hooks and how do I use them?"
- "How can I implement a battle system?"

## Deployment

This application is configured for deployment on Railway:

1. Fork this repository
2. Create a new Railway project
3. Connect your GitHub repository
4. Add the required environment variables
5. Deploy!

## Required Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_SERVICE_KEY`: Your Supabase service key
- `LLM_MODEL`: The OpenAI model to use (default: gpt-4o-mini)
- `PORT`: Automatically set by Railway

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - feel free to use this project as you wish.

## Acknowledgments

- Based on the [Crawl4AI Agent](https://github.com/coleam00/ottomator-agents/tree/main/crawl4AI-agent)
- Uses the excellent [RPGJS framework](https://rpgjs.dev)
- Built with [Pydantic AI](https://github.com/pydantic/pydantic-ai)
