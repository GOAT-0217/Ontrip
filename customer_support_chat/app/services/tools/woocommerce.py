import httpx
from langchain_core.tools import tool
from customer_support_chat.app.core.settings import get_settings
from customer_support_chat.app.core.logger import logger
from typing import List, Dict, Optional

settings = get_settings()

@tool
def search_products(query: str, limit: int = 10) -> str:
    """Search for products in WooCommerce based on a query.
    
    Args:
        query: The search query (e.g., product name, category).
        limit: Maximum number of products to return (default: 10).
        
    Returns:
        A formatted string containing product search results.
    """
    logger.info(f"WooCommerce search_products called with query: '{query}', limit: {limit}")
    
    if not settings.WOOCOMMERCE_API_URL or not settings.WOOCOMMERCE_CONSUMER_KEY or not settings.WOOCOMMERCE_CONSUMER_SECRET:
        return "WooCommerce API credentials are not configured."
    
    base_url = settings.WOOCOMMERCE_API_URL.rstrip('/')
    
    if not base_url.endswith('/wp-json/wc/v3'):
        if '/wp-json/wc/v3' not in base_url:
            url = f"{base_url}/wp-json/wc/v3/products"
        else:
            url = f"{base_url}/products"
    else:
        url = f"{base_url}/products"
    
    params = {
        "search": query,
        "per_page": min(limit, 100)
    }
    
    with httpx.Client(verify=False, timeout=30.0) as client:
        try:
            response = client.get(
                url,
                params=params,
                auth=httpx.BasicAuth(settings.WOOCOMMERCE_CONSUMER_KEY, settings.WOOCOMMERCE_CONSUMER_SECRET)
            )
            response.raise_for_status()
            products = response.json()
            
            results = []
            for product in products:
                name = product.get("name", "No Name")
                price = product.get("price", "N/A")
                desc = (product.get("short_description") or product.get("description", ""))[:100]
                results.append(f"Product: {name}, Price: {price}, Description: {desc}")
            
            if not results:
                return f"No products found for query: {query}"
            return "\n".join(results)
        except httpx.HTTPStatusError as e:
            return f"HTTP error occurred while searching products: {e} (Status: {e.response.status_code})"
        except httpx.TimeoutException:
            return "Timeout error while searching products. The WooCommerce server may be slow or unavailable."
        except httpx.ConnectError:
            return "Connection error while searching products. Check if WooCommerce server is running."
        except Exception as e:
            return f"An error occurred while searching products: {e}"

@tool
def search_orders(search_type: str, search_value: str, limit: int = 10) -> str:
    """Search for orders in WooCommerce based on specific criteria.
    
    Args:
        search_type: The type of search to perform. Must be one of: 'email', 'name', or 'id'.
        search_value: The value to search for based on the search_type.
        limit: Maximum number of orders to return (default: 10).
        
    Returns:
        A formatted string containing order search results.
    """
    logger.info(f"WooCommerce search_orders called with search_type: '{search_type}', search_value: '{search_value}'")
    
    if not settings.WOOCOMMERCE_API_URL or not settings.WOOCOMMERCE_CONSUMER_KEY or not settings.WOOCOMMERCE_CONSUMER_SECRET:
        return "WooCommerce API credentials are not configured."
    
    valid_search_types = ['email', 'name', 'id']
    if search_type not in valid_search_types:
        return f"Invalid search_type: {search_type}. Must be one of: {valid_search_types}"
    
    base_url = settings.WOOCOMMERCE_API_URL.rstrip('/')
    
    if not base_url.endswith('/wp-json/wc/v3'):
        if '/wp-json/wc/v3' not in base_url:
            url = f"{base_url}/wp-json/wc/v3/orders"
        else:
            url = f"{base_url}/orders"
    else:
        url = f"{base_url}/orders"
    
    params = {
        "per_page": min(limit, 100)
    }
    
    if search_type == 'email':
        params["customer_email"] = search_value
    elif search_type == 'name':
        params["search"] = search_value
    elif search_type == 'id':
        try:
            order_id = int(search_value)
            url = f"{url}/{order_id}"
            params = {}
        except ValueError:
            params["search"] = search_value
    
    with httpx.Client(verify=False, timeout=60.0) as client:
        try:
            response = client.get(
                url,
                params=params,
                auth=httpx.BasicAuth(settings.WOOCOMMERCE_CONSUMER_KEY, settings.WOOCOMMERCE_CONSUMER_SECRET)
            )
            response.raise_for_status()
            
            if search_type == 'id' and params == {}:
                order = response.json()
                orders = [order] if order else []
            else:
                orders = response.json()
            
            results = []
            for order in orders:
                order_id = order.get("id", "N/A")
                status = order.get("status", "N/A")
                total = order.get("total", "N/A")
                currency = order.get("currency", "N/A")
                date_created = order.get("date_created", "N/A")
                billing = order.get("billing", {})
                name = f"{billing.get('first_name', '')} {billing.get('last_name', '')}".strip()
                email = billing.get("email", "N/A")
                results.append(f"Order #{order_id}: Status: {status}, Total: {total} {currency}, Customer: {name}, Email: {email}, Date: {date_created}")
            
            if not results:
                return f"No orders found for {search_type} search with value '{search_value}'."
            return "\n".join(results)
        except httpx.HTTPStatusError as e:
            return f"HTTP error occurred while searching orders: {e} (Status: {e.response.status_code})"
        except httpx.TimeoutException:
            return "Timeout error while searching orders. The WooCommerce server may be slow or unavailable."
        except httpx.ConnectError:
            return "Connection error while searching orders. Check if WooCommerce server is running."
        except Exception as e:
            return f"An error occurred while searching orders: {e}"
