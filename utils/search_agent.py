import requests
import pandas as pd
from transformers import pipeline

# Initialize Hugging Face pipeline for text summarization or question answering
# You can choose other models such as `bert-large-uncased` for question answering or `t5-small` for summarization
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def search_web(query):
    """Use SerpAPI or similar API to perform web search."""
    api_key = "YOUR_SERPAPI_KEY"
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    response = requests.get(url)
    return response.json().get("organic_results", [])

def extract_info_from_text(text, prompt):
    """Extract relevant information from web search results using Hugging Face transformers."""
    # Combine the prompt with the search result text
    context = text
    # Use Hugging Face's pipeline for question answering or summarization
    result = qa_pipeline(question=prompt, context=context)
    return result['answer']  # Return the extracted answer

def search_web_and_extract_info(data, column, query_template):
    """Perform searches and extract information for each entity in the selected column."""
    results = []
    for entity in data[column]:
        query = query_template.replace("{Company}", entity)
        search_results = search_web(query)
        context_text = " ".join([res["snippet"] for res in search_results])  # Combine snippets
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
