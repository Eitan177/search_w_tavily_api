import streamlit as st
import requests
import random
import google.generativeai as genai

# Configure Gemini API key
GEMINI_API_KEY = st.secrets["GEMINI_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# Store Tavily API keys securely in Streamlit secrets
API_KEYS = [
    st.secrets["TAVILY_KEY_1"],
    st.secrets["TAVILY_KEY_2"],
    st.secrets["TAVILY_KEY_3"],
    st.secrets["TAVILY_KEY_4"],
    st.secrets["TAVILY_KEY_5"],
    st.secrets["TAVILY_KEY_6"]
]
TAVILY_URL = "https://api.tavily.com/search"

st.title("Variant Clinical Significance Search (Tavily API + Gemini Summary)")

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
    payload = {"query": query, "search_depth": "advanced"}  # switched to advanced for better results
    response = requests.post(TAVILY_URL, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        st.session_state.cache[query] = result
        return result
    else:
        return {"error": f"Request failed with status {response.status_code}"}

# Function to create integrated summary with Gemini and multiple fallbacks
def summarize_with_gemini(snippets, variant_name):
    if not snippets:
        return f"No relevant snippets found for {variant_name}."
    
    prompt = f"Summarize the following search results about {variant_name} into a concise clinical interpretation, focusing on clinical significance, pathogenicity classification (ACMG if available), and evidence sources.\n\n{' '.join(snippets)}"
    models_to_try = [
        "gemini-2.0-flash",
        "gemini-2.0-pro",
        "gemini-1.5-flash",
        "gemini-pro"
    ]

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and hasattr(response, 'text'):
                return response.text
        except Exception as e:
            st.warning(f"Model {model_name} failed: {e}. Trying next model...")
    
    return f"Unable to generate summary for {variant_name} after trying all backup models."

# Search all variants and display integrated summaries
if st.button("Search Clinical Significance"):
    for variant in st.session_state.variants:
        if variant.strip():
            query = f"Please investigate the clinical significance of {variant}"
            st.write(f"**Searching for:** {variant}")
            result = search_tavily(query)
            if "error" in result:
                st.error(result["error"])
            else:
                snippets = [r.get("snippet", "") for r in result.get("results", []) if r.get("snippet")]
                summary = summarize_with_gemini(snippets, variant)
                st.markdown(f"**Gemini Clinical Summary for {variant}:**\n\n{summary}")
                st.write("---")
