import os
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from serpapi import GoogleSearch

# Load environment variables from a .env file
load_dotenv()

app = Flask(__name__)

# Retrieve API keys from environment variables
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")  # Optional fallback provider


def perform_serp_search(query: str, num_results: int = 3) -> str:
    """Fetch search result snippets using SerpApi."""
    if not SERPAPI_KEY:
        return ""

    try:
        search = GoogleSearch({"q": query, "api_key": SERPAPI_KEY, "num": num_results})
        results = search.get_dict()
        organic = results.get("organic_results", [])

        snippets = []
        for res in organic:
            title = res.get("title", "")
            snippet = res.get("snippet", "")
            snippets.append(f"Title: {title}\nSnippet: {snippet}")

        return "\n\n".join(snippets)
    except Exception as e:
        print(f"SerpApi Error: {e}")
        return ""


def perform_tavily_search(query: str) -> str:
    """Fallback search function using Tavily API if SerpApi produces no context."""
    if not TAVILY_KEY:
        return ""

    try:
        url = "https://api.tavily.com/search"
        payload = {"api_key": TAVILY_KEY, "query": query, "max_results": 3}
        response = requests.post(url, json=payload, timeout=10)
        data = response.json()

        results = data.get("results", [])
        snippets = [
            f"Title: {item.get('title')}\nSnippet: {item.get('content')}"
            for item in results
        ]
        return "\n\n".join(snippets)
    except Exception as e:
        print(f"Tavily Error: {e}")
        return ""


def generate_openrouter_response(
    prompt: str, context: str, model: str = "openrouter/free"
) -> str:
    """Send search-augmented prompt to OpenRouter API."""
    if not OPENROUTER_KEY:
        return "Error: OpenRouter API Key is missing."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Search AI Agent",
    }

    # Construct system and user prompt ensuring natural tone and grounding
    system_prompt = (
        "You are an AI assistant. Answer the user's request using the search context provided below. "
        "Keep your tone direct, clear, and human-like. Avoid robotic filler words."
    )

    user_message = (
        f"Search Context:\n{context}\n\nUser Question:\n{prompt}"
        if context
        else prompt
    )

    payload = {
        "model": model,  # Uses free model routing or specific model ID
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        data = response.json()

        if response.status_code == 200:
            return data["choices"][0]["message"]["content"]
        else:
            return f"OpenRouter API Error: {data.get('error', {}).get('message', 'Unknown error')}"

    except Exception as e:
        return f"Request failed: {str(e)}"


@app.route("/generate", methods=["POST"])
def search_and_generate():
    """Main API endpoint to handle request, fetch web context, and query OpenRouter."""
    data = request.get_json() or {}
    query = data.get("prompt", "")

    if not query:
        return jsonify({"error": "Prompt parameter is required."}), 400

    # Step 1: Query SerpApi for web results
    search_context = perform_serp_search(query)

    # Step 2: Fallback to alternative provider if SerpApi context is empty
    if not search_context:
        search_context = perform_tavily_search(query)

    # Step 3: Pass context + prompt to OpenRouter
    llm_output = generate_openrouter_response(query, search_context)

    return jsonify(
        {
            "query": query,
            "context_found": bool(search_context),
            "response": llm_output,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
