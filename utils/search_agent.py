import requests
import pandas as pd
from transformers import pipeline

# Initialize Hugging Face pipeline for question answering
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def search_web(query):
    """Use SerpAPI or similar API to perform web search."""
    api_key = "YOUR_SERPAPI_KEY"
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("organic_results", [])
    except requests.exceptions.RequestException as e:
        print(f"Error in search_web: {e}")
        return []  # Return an empty list if there's an error

def extract_info_from_text(text, prompt):
    """Extract relevant information from web search results using Hugging Face transformers."""
    try:
        result = qa_pipeline(question=prompt, context=text)
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
        context_text = " ".join([res.get("snippet", "") for res in search_results if "snippet" in res])
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
