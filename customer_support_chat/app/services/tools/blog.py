# customer_support_chat/app/services/tools/blog.py

import httpx
import urllib.parse
from langchain_core.tools import tool
from customer_support_chat.app.core.settings import get_settings
from typing import List, Dict

settings = get_settings()

@tool
def search_blog_posts(keyword: str, limit: int = 5) -> str:
    """Search for blog posts based on a keyword.
    
    Args:
        keyword: The keyword to search for in blog posts.
        limit: Maximum number of posts to return (default: 5).
        
    Returns:
        A formatted string containing blog post results.
    """
    if not settings.BLOG_SEARCH_API_URL:
        return "Blog search API URL is not configured."
    
    encoded_keyword = urllib.parse.quote(keyword)
    url = f"{settings.BLOG_SEARCH_API_URL}?search={encoded_keyword}"
    
    auth = None
    if settings.WOOCOMMERCE_CONSUMER_KEY and settings.WOOCOMMERCE_CONSUMER_SECRET:
        auth = httpx.BasicAuth(settings.WOOCOMMERCE_CONSUMER_KEY, settings.WOOCOMMERCE_CONSUMER_SECRET)
    
    with httpx.Client() as client:
        try:
            response = client.get(
                url,
                auth=auth
            )
            response.raise_for_status()
            posts = response.json()
            
            results = []
            for post in posts[:limit]:
                title = post.get("title", {}).get("rendered", "No Title")
                excerpt = post.get("excerpt", {}).get("rendered", "")[:200]
                link = post.get("link", "")
                date = post.get("date", "")
                results.append(f"Title: {title}\nDate: {date}\nLink: {link}\nExcerpt: {excerpt}")
            
            if not results:
                return f"No blog posts found for keyword: {keyword}"
            return "\n\n".join(results)
        except httpx.HTTPStatusError as e:
            return f"HTTP error occurred while searching blog posts: {e}"
        except Exception as e:
            return f"An error occurred while searching blog posts: {e}"