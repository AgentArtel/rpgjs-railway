import os
import sys
import json
import asyncio
import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import re

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]
    version: str
    heading: str = None
    subheading: str = None
    code_blocks: List[str] = None
    source: str = "rpgjs_docs"
    last_updated: str = datetime.now(timezone.utc).isoformat()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove multiple newlines and spaces
    text = text.replace('\n\n', '\n')
    text = text.replace('  ', ' ')
    # Remove code block markers but keep the code
    text = text.replace('```', '')
    return text.strip()

def normalize_url(url: str, base_url: str = None) -> str:
    """Normalize URL by removing trailing slashes, .html extensions, and handling relative paths."""
    if not url:
        return None
        
    # Handle relative URLs
    if url.startswith('/'):
        if base_url:
            parsed_base = urlparse(base_url)
            url = f"{parsed_base.scheme}://{parsed_base.netloc}{url}"
    elif not url.startswith(('http://', 'https://')):
        if base_url:
            if base_url.endswith('/'):
                url = base_url + url
            else:
                url = base_url + '/' + url
                
    # Remove .html extension
    if url.endswith('.html'):
        url = url[:-5]
        
    # Remove trailing slash
    if url.endswith('/'):
        url = url[:-1]
        
    return url

def is_valid_doc_url(url: str) -> bool:
    """Check if URL is a valid documentation URL."""
    if not url:
        return False
        
    # Skip external links, anchors, etc.
    if url.startswith(('mailto:', 'tel:', 'javascript:', '#')):
        return False
        
    # Must be from rpgjs.dev domains
    parsed = urlparse(url)
    if not any(domain in parsed.netloc for domain in ['rpgjs.dev', 'docs.rpgjs.dev']):
        return False
        
    # Skip asset files
    if parsed.path.endswith(('.jpg', '.png', '.gif', '.css', '.js')):
        return False
        
    return True

def check_sitemap(base_url: str) -> set[str]:
    """Try to get URLs from sitemap.xml if available."""
    urls = set()
    try:
        # Try common sitemap locations
        sitemap_urls = [
            f"{base_url}/sitemap.xml",
            f"{base_url}/sitemap_index.xml",
            f"{base_url}/sitemap"
        ]
        
        for sitemap_url in sitemap_urls:
            try:
                response = requests.get(sitemap_url)
                if response.status_code == 200:
                    # Parse XML
                    root = ElementTree.fromstring(response.content)
                    # Handle both sitemap and sitemapindex
                    for loc in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                        url = normalize_url(loc.text, base_url)
                        if is_valid_doc_url(url):
                            urls.add(url)
                            
                            # If this is a sitemap index, fetch the child sitemaps
                            if 'sitemap' in loc.text:
                                child_urls = check_sitemap(loc.text)
                                urls.update(child_urls)
            except Exception as e:
                print(f"Error checking sitemap {sitemap_url}: {e}")
                
    except Exception as e:
        print(f"Error checking sitemaps for {base_url}: {e}")
        
    return urls

def get_doc_urls(base_url: str, visited: set[str] = None, depth: int = 0) -> set[str]:
    """Get all documentation page URLs recursively."""
    if visited is None:
        visited = set()
        
    # Limit recursion depth
    if depth > 5:
        return set()
        
    urls = set()
    
    # Skip if already visited
    if base_url in visited:
        return urls
        
    visited.add(base_url)
    print(f"Crawling {base_url} (depth {depth})")
    
    try:
        # Then crawl the page
        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links
        for link in soup.find_all(['a', 'link'], href=True):
            href = link['href']
            url = normalize_url(href, base_url)
            
            if not url or not is_valid_doc_url(url):
                continue
                
            urls.add(url)
            
            # Recursively crawl if not visited
            if url not in visited:
                sub_urls = get_doc_urls(url, visited, depth + 1)
                urls.update(sub_urls)
                
        # Look for specific documentation patterns
        if depth == 0:  # Only check patterns at root level
            patterns = [
                '/guide/',
                '/api/',
                '/examples/',
                '/tutorials/',
                '/docs/',
                '/reference/',
                '/manual/',
                '/getting-started/',
                '/advanced/',
                '/plugins/',
                '/components/'
            ]
            
            for pattern in patterns:
                if pattern in base_url:
                    # Check numbered pages
                    for i in range(1, 10):
                        page_url = normalize_url(f"{base_url}/{i}", base_url)
                        if page_url and page_url not in visited:
                            try:
                                response = requests.get(page_url)
                                if response.status_code == 200:
                                    urls.add(page_url)
                                    sub_urls = get_doc_urls(page_url, visited, depth + 1)
                                    urls.update(sub_urls)
                            except Exception as e:
                                print(f"Error checking numbered page {page_url}: {e}")
                                
                    # Check common subsections
                    subsections = ['basic', 'advanced', 'examples', 'plugins', 'api', 'guide']
                    for section in subsections:
                        section_url = normalize_url(f"{base_url}/{section}", base_url)
                        if section_url and section_url not in visited:
                            try:
                                response = requests.get(section_url)
                                if response.status_code == 200:
                                    urls.add(section_url)
                                    sub_urls = get_doc_urls(section_url, visited, depth + 1)
                                    urls.update(sub_urls)
                            except Exception as e:
                                print(f"Error checking subsection {section_url}: {e}")
                            
    except Exception as e:
        print(f"Error fetching URLs from {base_url}: {e}")
        
    return urls

def get_rpgjs_docs_urls() -> List[str]:
    """Get URLs from RPGJS documentation."""
    base_urls = [
        "https://docs.rpgjs.dev",
        "https://rpgjs.dev/guide",
        "https://rpgjs.dev/api"
    ]
    
    all_urls = set()
    visited = set()
    
    for base_url in base_urls:
        try:
            urls = get_doc_urls(base_url, visited)
            all_urls.update(urls)
            print(f"Found {len(urls)} URLs from {base_url}")
        except Exception as e:
            print(f"Error fetching URLs from {base_url}: {e}")
    
    # Sort URLs for consistent ordering
    sorted_urls = sorted(list(all_urls))
    print(f"Total unique URLs found: {len(sorted_urls)}")
    return sorted_urls

def extract_code_blocks(text: str) -> tuple[str, List[str]]:
    """Extract code blocks from text and return cleaned text and list of code blocks."""
    code_blocks = []
    cleaned_text = text
    
    # Find all code blocks (```...```)
    code_pattern = r'```(?:\w+)?\n(.*?)```'
    matches = re.finditer(code_pattern, text, re.DOTALL)
    
    for match in matches:
        code_block = match.group(1).strip()
        code_blocks.append(code_block)
        # Replace code block with placeholder in cleaned text
        cleaned_text = cleaned_text.replace(match.group(0), f'[Code Block {len(code_blocks)}]')
    
    return cleaned_text, code_blocks

def extract_headings(soup: BeautifulSoup, url: str) -> tuple[str, str]:
    """Extract heading and subheading from HTML."""
    heading = None
    subheading = None
    
    # Try to find main heading
    h1_tags = soup.find_all('h1')
    if h1_tags:
        heading = h1_tags[0].get_text().strip()
        
        # Look for relevant subheadings after the main heading
        for h1 in h1_tags:
            next_tag = h1.find_next(['h2', 'h3'])
            if next_tag:
                subheading = next_tag.get_text().strip()
                break
    
    # If no h1, try h2 as main heading
    if not heading:
        h2_tags = soup.find_all('h2')
        if h2_tags:
            heading = h2_tags[0].get_text().strip()
            
            # Look for relevant subheadings
            if len(h2_tags) > 1:
                subheading = h2_tags[1].get_text().strip()
    
    # If still no heading, try to extract from title or URL
    if not heading:
        title_tag = soup.find('title')
        if title_tag:
            heading = title_tag.get_text().strip()
        else:
            # Extract from URL as last resort
            path = urlparse(url).path
            heading = path.split('/')[-1].replace('-', ' ').title()
    
    return heading, subheading

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def process_chunk(chunk: str, chunk_number: int, url: str, heading: str = None, subheading: str = None) -> ProcessedChunk:
    """Process a single chunk of text."""
    try:
        # Extract version from URL or content
        version_match = re.search(r'v(\d+\.\d+\.\d+)', url)
        version = version_match.group(1) if version_match else "latest"
        
        # Extract code blocks
        cleaned_text, code_blocks = extract_code_blocks(chunk)
        
        # Get title and summary
        extracted = await get_title_and_summary(cleaned_text, url)
        title = extracted['title']
        summary = extracted['summary']
        
        # Get embedding for cleaned text
        embedding = await get_embedding(cleaned_text)
        
        # Create metadata
        metadata = {
            "version": version,
            "chunk_size": len(chunk),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "url_path": urlparse(url).path,
            "has_code_blocks": len(code_blocks) > 0
        }
        
        # Create processed chunk
        processed = ProcessedChunk(
            url=url,
            chunk_number=chunk_number,
            title=title,
            summary=summary,
            content=cleaned_text,
            metadata=metadata,
            embedding=embedding,
            version=version,
            heading=heading,
            subheading=subheading,
            code_blocks=code_blocks if code_blocks else []
        )
        
        return processed
    except Exception as e:
        print(f"Error processing chunk: {e}")
        raise

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding,
            "version": chunk.version,
            "source": chunk.source,
            "heading": chunk.heading,
            "subheading": chunk.subheading,
            "code_blocks": chunk.code_blocks,
            "last_updated": chunk.last_updated
        }
        
        result = supabase.table("rpgjs_documentation").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk into database: {e}")
        return None

async def process_and_store_document(url: str, html: str):
    """Process a document and store its chunks in parallel."""
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract headings
        heading, subheading = extract_headings(soup, url)
        
        # Get main content
        # First try to find the main content container
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.content', '.main-content', '#content', '#main']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body')
        
        # Remove navigation, footer, and other non-content elements
        for element in main_content.find_all(['nav', 'footer', 'header', 'aside']):
            element.decompose()
            
        # Extract content while preserving some structure
        content_parts = []
        
        # Process headings and their content
        for tag in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'pre', 'code', 'ul', 'ol', 'li']):
            if tag.name.startswith('h'):
                # Add heading with proper markdown
                level = int(tag.name[1])
                content_parts.append('\n' + ('#' * level) + ' ' + tag.get_text().strip() + '\n')
            elif tag.name == 'pre' or tag.name == 'code':
                # Preserve code blocks
                code = tag.get_text().strip()
                if code:
                    content_parts.append('\n```\n' + code + '\n```\n')
            elif tag.name in ['ul', 'ol']:
                # Handle lists
                for li in tag.find_all('li', recursive=False):
                    content_parts.append('- ' + li.get_text().strip())
            else:
                # Regular paragraphs and other content
                text = tag.get_text().strip()
                if text:
                    content_parts.append(text + '\n')
        
        content = '\n'.join(content_parts)
        content = clean_text(content)
        
        # Split into chunks
        chunks = chunk_text(content)
        
        # Process chunks in parallel
        tasks = []
        for i, chunk in enumerate(chunks):
            task = process_chunk(chunk, i, url, heading, subheading)
            tasks.append(task)
        
        processed_chunks = await asyncio.gather(*tasks)
        
        # Store chunks
        for chunk in processed_chunks:
            await insert_chunk(chunk)
            
    except Exception as e:
        print(f"Error processing document {url}: {e}")

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.html)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

async def main():
    # Get URLs from RPGJS documentation
    urls = get_rpgjs_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())
