import requests
import pandas as pd
from google.cloud import language_v1
from google.cloud.language_v1 import enums

# Function to perform web search using SerpAPI (or any other search API)
def search_web(query):
    """Use SerpAPI or similar API to perform web search."""
    api_key = "YOUR_SERPAPI_KEY"
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    response = requests.get(url)
    return response.json().get("organic_results", [])

# Function to extract relevant information using Google Cloud Natural Language API
def extract_info_from_text(text, prompt):
    """Extract relevant information from web search results using Google Cloud NLP API."""
    # Set up Google Cloud Language API client
    client = language_v1.LanguageServiceClient()

    # Prepare the text document
    document = language_v1.Document(content=text, type_=enums.Document.Type.PLAIN_TEXT)

    # Use the API to analyze the entities in the text
    response = client.analyze_entities(document=document)
    
    # Extract entities and return their names and types
    entities = []
    for entity in response.entities:
        entities.append({
            "name": entity.name,
            "type": enums.Entity.Type(entity.type_).name,
            "salience": entity.salience  # Salience indicates the relevance of the entity in the context
        })
    
    # If you want to return all entity details (including types, salience, etc.)
    return entities

# Function to search the web and extract information for each entity in the selected column
def search_web_and_extract_info(data, column, query_template):
    """Perform searches and extract information for each entity in the selected column."""
    results = []
    for entity in data[column]:
        query = query_template.replace("{Company}", entity)
        search_results = search_web(query)
        
        # Combine snippets of the search results to form context text
        context_text = " ".join([res["snippet"] for res in search_results])
        
        # Extract information from the combined search results using Google Cloud NLP API
        extracted_info = extract_info_from_text(context_text, f"Extract info for {entity}: ")
        
        # Append entity and extracted information to results
        results.append((entity, extracted_info))
    
    # Return the results as a DataFrame
    return pd.DataFrame(results, columns=[column, "Extracted Info"])

