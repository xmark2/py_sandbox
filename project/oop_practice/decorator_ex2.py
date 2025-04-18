import requests
from functools import wraps

def evaluate_url_response(func, url, *args, **kwargs):
    """Evaluate the response returned by the function."""
    try:
        response = func(url, *args, **kwargs)
        if response and response.content:  # Check if data is returned
            print("Data retrieved successfully!")
            return response.content
        else:
            print("No data received, retrying...")
    except requests.RequestException as e:
        print(f"Error occurred: {e}, retrying...")
    return None  # Explicitly return None if no data or an error occurs

def retry_on_failure(max_retries=3):
    """Decorator to retry a function multiple times on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(url, *args, **kwargs):
            for attempt in range(max_retries):
                print(f"Attempt {attempt + 1} for URL: {url}")
                resp = evaluate_url_response(func, url, *args, **kwargs)
                if resp:
                    return resp
            print("Failed to retrieve data after maximum retries.")
            return None
        return wrapper
    return decorator

@retry_on_failure(max_retries=3)
def fetch_url_data(url):
    """Fetch data from the specified URL."""
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad HTTP status codes
    return response

if __name__ == "__main__":
    # Example usage
    url = "https://jsonplaceholder.typicode.com/posts/1"  # Sample valid URL
    data = fetch_url_data(url)
    if data:
        print(f"Data received: {data[:50]}...")  # Display first 50 characters
    else:
        print("Failed to fetch data from the URL.")