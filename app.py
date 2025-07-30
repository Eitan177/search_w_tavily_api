import streamlit as st
import requests
import random

# Store API keys securely in Streamlit secrets
API_KEYS = [
    st.secrets["TAVILY_KEY_1"],
    st.secrets["TAVILY_KEY_2"],
    st.secrets["TAVILY_KEY_3"],
    st.secrets["TAVILY_KEY_4"],
    st.secrets["TAVILY_KEY_5"],
    st.secrets["TAVILY_KEY_6"]
]
TAVILY_URL = "https://api.tavily.com/search"

st.title("Variant Clinical Significance Search (Tavily API)")

# Maintain a list of variants using session state
if "variants" not in st.session_state:
    st.session_state.variants = [""]

# Cache for search results to avoid repeated API calls
if "cache" not in st.session_state:
    st.session_state.cache = {}

# Add new variant input
def add_variant():
    st.session_state.variants.append("")

# Variant input fields
for i, variant in enumerate(st.session_state.variants):
    st.session_state.variants[i] = st.text_input(f"Variant {i+1}", value=variant, key=f"variant_{i}")

# Add another variant button
if st.button("Add another variant"):
    add_variant()

# Search function with caching
def search_tavily(query):
    if query in st.session_state.cache:
        return st.session_state.cache[query]

    api_key = random.choice(API_KEYS)  # Randomly select a key each time
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"query": query, "search_depth": "basic"}
    response = requests.post(TAVILY_URL, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        st.session_state.cache[query] = result
        return result
    else:
        return {"error": f"Request failed with status {response.status_code}"}

# Search all variants
if st.button("Search Clinical Significance"):
    for variant in st.session_state.variants:
        if variant.strip():
            query = f"Please investigate the clinical significance of {variant}"
            st.write(f"**Searching for:** {variant}")
            result = search_tavily(query)
            if "error" in result:
                st.error(result["error"])
            else:
                for r in result.get("results", []):
                    st.markdown(f"- [{r.get('title')}]({r.get('url')})")
                    st.write(r.get("snippet"))
                st.write("---")

