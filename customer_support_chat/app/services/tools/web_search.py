import os
import re
import json
import requests
from typing import List, Dict, Optional
from langchain_core.tools import tool
from customer_support_chat.app.core.logger import logger
from customer_support_chat.app.core.settings import get_settings

settings = get_settings()

_EMOJI_PATTERN = re.compile(r'[\U00010000-\U0010ffff]')

def _remove_emoji(text: str) -> str:
    return _EMOJI_PATTERN.sub('', text)

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for information using multiple search engines.
    
    This tool searches for real-time information from the internet,
    useful for finding hotel recommendations, travel tips, local attractions,
    restaurant reviews, and other up-to-date information.
    
    Args:
        query: The search query (e.g., "上海外滩附近推荐酒店", "best hotels near Shanghai Bund")
        max_results: Maximum number of results to return (default: 5)
        
    Returns:
        A formatted text string containing search results.
    """
    try:
        results = _search_with_tavily(query, max_results)
        if not results:
            results = _search_with_bing(query, max_results)
        
        if not results:
            return f"No search results found for '{query}'. Please try a different query."
        
        # Format results as clean text (avoid large JSON that may cause API errors)
        formatted = _format_search_results(results, query)
        return formatted
        
    except Exception as e:
        logger.error(f"web_search error: {e}")
        return f"Search encountered an error: {str(e)}. Please try again or rephrase your query."


def _format_search_results(results: List[Dict], query: str) -> str:
    lines = [f"Search Results for '{query}':\n"]
    
    for i, item in enumerate(results[:5], 1):
        title = _remove_emoji(item.get('title', 'No Title').strip())
        snippet = _remove_emoji(item.get('snippet', '').strip())
        
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        
        lines.append(f"{i}. {title}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")
    
    return "\n".join(lines)


def _search_with_tavily(query: str, max_results: int = 5) -> Optional[List[Dict]]:
    """Search using Tavily API (recommended, reliable)."""
    api_key = getattr(settings, 'TAVILY_API_KEY', None) or os.getenv('TAVILY_API_KEY')
    
    if not api_key:
        logger.debug("TAVILY_API_KEY not configured")
        return None
    
    try:
        response = requests.post(
            'https://api.tavily.com/search',
            json={
                'api_key': api_key,
                'query': query,
                'max_results': max_results,
                'include_answer': True,
                'include_raw_content': False
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            if 'answer' in data and data['answer']:
                results.append({
                    'title': 'AI Summary',
                    'url': '',
                    'snippet': data['answer'],
                    'source': 'tavily-ai'
                })
            
            for item in data.get('results', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'snippet': item.get('content', '')[:300],
                    'source': 'tavily'
                })
            
            logger.info(f"Tavily search returned {len(results)} results for: {query[:50]}...")
            return results[:max_results]
        else:
            logger.warning(f"Tavily API error: {response.status_code}")
            return None
            
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return None


def _search_with_bing(query: str, max_results: int = 5) -> Optional[List[Dict]]:
    """Search using Bing Web Search API or fallback method."""
    bing_key = getattr(settings, 'BING_API_KEY', None) or os.getenv('BING_API_KEY')
    
    if bing_key:
        try:
            response = requests.get(
                'https://api.bing.microsoft.com/v7.0/search',
                params={
                    'q': query,
                    'count': max_results,
                    'mkt': 'zh-CN'
                },
                headers={'Ocp-Apim-Subscription-Key': bing_key},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('webPages', {}).get('value', []):
                    results.append({
                        'title': item.get('name', ''),
                        'url': item.get('url', ''),
                        'snippet': item.get('snippet', ''),
                        'source': 'bing'
                    })
                
                logger.info(f"Bing search returned {len(results)} results")
                return results
                
        except Exception as e:
            logger.warning(f"Bing API search failed: {e}")
    
    return _search_with_requests_fallback(query, max_results)


def _search_with_requests_fallback(query: str, max_results: int = 5) -> Optional[List[Dict]]:
    """Fallback: Use multiple search engines with direct HTTP requests."""
    
    # Try Bing first (usually accessible in China)
    results = _try_bing_direct(query, max_results)
    if results:
        return results
    
    # Try DuckDuckGo HTML
    results = _try_duckduckgo_html(query, max_results)
    if results:
        return results
    
    return [{
        'title': 'Search Unavailable',
        'url': '',
        'snippet': f'Unable to search for "{query}". Please configure TAVILY_API_KEY in .env for reliable search.',
        'source': 'error'
    }]


def _try_bing_direct(query: str, max_results: int = 5) -> Optional[List[Dict]]:
    """Try searching via Bing directly."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
        }
        
        encoded_query = requests.utils.quote(query)
        url = f'https://www.bing.com/search?q={encoded_query}&count={max_results}&setlang=zh-Hans'
        
        response = requests.get(url, headers=headers, timeout=8)
        
        if response.status_code == 200:
            import re
            text = response.text
            
            results = []
            
            # Pattern to extract search results from Bing
            patterns = [
                r'<li class="b_algo"[^>]*>.*?<h2><a[^>]*href="([^"]*)"[^>]*>(.*?)</a></h2>.*?<p[^>]*>(.*?)</p>',
                r'<div class="b_caption"[^>]*>.*?<p[^>]*>(.*?)</p>',
            ]
            
            # Simple extraction
            b_algo_blocks = re.findall(r'<li class="b_algo"(.*?)</li>', text, re.DOTALL | re.IGNORECASE)
            
            for block in b_algo_blocks[:max_results]:
                # Try multiple patterns to extract title
                title = ''
                
                # Pattern 1: h2 > a tag
                title_match = re.search(r'<h2[^>]*>.*?<a[^>]*>(.*?)</a>', block, re.DOTALL | re.IGNORECASE)
                if title_match:
                    title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()
                
                # Pattern 2: Any strong or b tag with text
                if not title:
                    title_match = re.search(r'<(?:strong|b)[^>]*>([^<]+)</(?:strong|b)>', block, re.IGNORECASE)
                    if title_match:
                        title = title_match.group(1).strip()
                
                # Pattern 3: First caption
                if not title:
                    caption_match = re.search(r'class="b_caption"[^>]*>(.*?)</p>', block, re.DOTALL | re.IGNORECASE)
                    if caption_match:
                        caption_text = re.sub(r'<[^>]+>', '', caption_match.group(1)).strip()
                        if len(caption_text) > 10:
                            title = caption_text[:80]
                
                url_match = re.search(r'href="(https?://[^"]+)"', block)
                snippet_match = re.search(r'<p[^>]*>(.*?)</p>', block, re.DOTALL | re.IGNORECASE)
                
                url = url_match.group(1) if url_match else ''
                snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip() if snippet_match else ''
                
                # Use snippet as title if no title found
                if not title and snippet:
                    title = snippet[:80]
                
                if title or snippet:
                    results.append({
                        'title': title[:100] if title else 'Search Result',
                        'url': url[:200],
                        'snippet': snippet[:300] if snippet else '',
                        'source': 'bing-direct'
                    })
            
            if results:
                logger.info(f"Bing direct search returned {len(results)} results")
                return results
                
    except Exception as e:
        logger.warning(f"Bing direct search failed: {e}")
    
    return None


def _try_duckduckgo_html(query: str, max_results: int = 5) -> Optional[List[Dict]]:
    """Try searching via DuckDuckGo HTML version."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        encoded_query = requests.utils.quote(query)
        url = f'https://html.duckduckgo.com/html/?q={encoded_query}'
        
        response = requests.get(url, headers=headers, timeout=8)
        
        if response.status_code == 200:
            import re
            text = response.text
            
            results = []
            
            # Extract results from DDG HTML
            result_blocks = re.findall(r'<div class="result"[^>]*>(.*?)</div>\s*</div>', text, re.DOTALL | re.IGNORECASE)
            
            for block in result_blocks[:max_results]:
                title_match = re.search(r'class="result__title"[^>]*>.*?<a[^>]*class="result__a"[^>]*>(.*?)</a>', block, re.DOTALL)
                snippet_match = re.search(r'class="result__snippet"[^>]*>(.*?)</[at]', block, re.DOTALL)
                url_match = re.search(r'href="(.*?)"', block)
                
                title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip() if title_match else ''
                snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip() if snippet_match else ''
                
                if title:
                    results.append({
                        'title': title[:100],
                        'url': '',
                        'snippet': snippet[:300],
                        'source': 'duckduckgo-html'
                    })
            
            if results:
                logger.info(f"DuckDuckGo HTML returned {len(results)} results")
                return results
                
    except Exception as e:
        logger.warning(f"DuckDuckGo HTML failed: {e}")
    
    return None
