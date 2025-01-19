# Crawl4AI Agent - Railway Deployment

This is a Railway deployment of the [Crawl4AI Agent](https://github.com/coleam00/ottomator-agents/tree/main/crawl4AI-agent).

## Required Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_SERVICE_KEY`: Your Supabase service key
- `PORT`: Automatically set by Railway

## Deployment

This application is configured to deploy on Railway with:
- Python 3.11
- Streamlit web interface
- Supabase database
- OpenAI API integration

The application will automatically use the PORT environment variable provided by Railway.
