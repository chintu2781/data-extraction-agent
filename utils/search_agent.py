import requests
import pandas as pd
from transformers import pipeline
import os
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Initialize Hugging Face pipeline for question answering
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def search_web(query):
    """Use SerpAPI or similar API to perform web search."""
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("Error: SERPAPI_KEY environment variable not found.")
        return []
    print(f"Using API key: {api_key[:4]}****")
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    try:
        response = session.get(url)
        response.raise_for_status()
        results = response.json().get("organic_results", [])
        if not results:
            print("No organic results found in API response.")
        return results
    except requests.exceptions.RequestException as e:
        print(f"Error in search_web: {e}")
        return []

def extract_info_from_text(text, prompt):
    """Extract relevant information from web search results using Hugging Face transformers."""
    try:
        print(f"Extracting with prompt: '{prompt}' and context: '{text[:200]}...'")  # Print first 200 characters of text
        result = qa_pipeline(question=prompt, context=text)
        print("Extraction result:", result)
        return result.get('answer', 'No answer found')
    except Exception as e:
        print(f"Error in extract_info_from_text: {e}")
        return "Error in extraction"

def search_web_and_extract_info(data, column, query_template):
    """Perform searches and extract information for each entity in the selected column."""
    results = []
    for entity in data[column]:
        query = query_template.replace("{Company}", entity)
        search_results = search_web(query)

        # Debug print to check if search results are coming through
        print(f"Search results for {entity}: {search_results}")
        
        if not search_results:
            print(f"No search results found for {entity}")
            results.append((entity, "No information found"))
            continue

        context_text = " ".join([res.get("snippet", "") for res in search_results if "snippet" in res])

        # Debug print for context_text
        print(f"context_text for {entity}: '{context_text}'")
        
        if not context_text:
            print(f"No context text available for {entity}")
            results.append((entity, "No information found"))
            continue

        result = extract_info_from_text(context_text, f"Extract information for {entity}")
        results.append((entity, result))

    return pd.DataFrame(results, columns=[column, "Extracted Info"])



# Example usage:
data = pd.DataFrame({
    'Company': ['Apple', 'Tesla', 'Microsoft']
})

column = 'Company'
query_template = "Search information about {Company}."

# Run searches and extract info
results_df = search_web_and_extract_info(data, column, query_template)
print(results_df)
