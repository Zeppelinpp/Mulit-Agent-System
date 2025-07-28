import os
from typing import List
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()


def fetch_content(urls: List[str], client: TavilyClient):
    content = client.extract(urls=urls)["results"]
    raw_contents = [
        (item["url"] + ":\n" + item["raw_content"][:1000]) for item in content
    ]
    return "\n".join(raw_contents)


def web_search(query: str):
    """Search Information from the web based on the query"""
    client = TavilyClient(api_key=os.getenv("TAVILY_KEY"))
    results = client.search(query=query, max_results=1)["results"]
    related_urls = [result["url"] for result in results]
    return fetch_content(related_urls, client)


if __name__ == "__main__":
    query = "What is the latest trend in AI?"
    print(web_search(query))
