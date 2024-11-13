import requests
import openai

def search_web(query):
    """Use SerpAPI or similar API to perform web search."""
    api_key = "YOUR_SERPAPI_KEY"
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    response = requests.get(url)
    return response.json().get("organic_results", [])

def extract_info_from_text(text, prompt):
    """Extract relevant information from web search results using OpenAI API."""
    openai.api_key = "YOUR_OPENAI_API_KEY"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt + text,
        max_tokens=100
    )
    return response['choices'][0]['text'].strip()

def search_web_and_extract_info(data, column, query_template):
    """Perform searches and extract information for each entity in the selected column."""
    results = []
    for entity in data[column]:
        query = query_template.replace("{Company}", entity)
        search_results = search_web(query)
        context_text = " ".join([res["snippet"] for res in search_results])
        result = extract_info_from_text(context_text, f"Extract info for {entity}: ")
        results.append((entity, result))
    return pd.DataFrame(results, columns=[column, "Extracted Info"])

