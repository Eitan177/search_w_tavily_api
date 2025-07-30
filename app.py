import streamlit as st
import requests
import random
import google.generativeai as genai

# Configure Gemini API key
# Make sure "GEMINI_KEY" is set in your Streamlit secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("Gemini API key not found. Please set `GEMINI_KEY` in your Streamlit secrets.")
    st.stop()


# Store Tavily API keys securely in Streamlit secrets
# Ensure these are also set in your Streamlit secrets
try:
    API_KEYS = [
        st.secrets["TAVILY_KEY_1"],
        st.secrets["TAVILY_KEY_2"],
        st.secrets["TAVILY_KEY_3"],
        st.secrets["TAVILY_KEY_4"],
        st.secrets["TAVILY_KEY_5"],
        st.secrets["TAVILY_KEY_6"]
    ]
except (KeyError, FileNotFoundError):
    st.error("Tavily API keys not found. Please set `TAVILY_KEY_1` through `TAVILY_KEY_6` in your Streamlit secrets.")
    st.stop()

TAVILY_URL = "https://api.tavily.com/search"

st.title("Variant Clinical Significance Search (Tavily API + Gemini Summary)")

# --- Session State Initialization ---
# Maintain a list of variants using session state
if "variants" not in st.session_state:
    st.session_state.variants = [""]

# Cache for search results to avoid repeated API calls
if "cache" not in st.session_state:
    st.session_state.cache = {}

# --- UI Components ---
# Add new variant input
def add_variant():
    st.session_state.variants.append("")

# Variant input fields
for i, variant in enumerate(st.session_state.variants):
    st.session_state.variants[i] = st.text_input(f"Variant {i+1}", value=variant, key=f"variant_{i}")

# Add another variant button
if st.button("Add another variant"):
    add_variant()

# --- API Functions ---
# Search function with caching
def search_tavily(query):
    """Performs a search using the Tavily API and caches the result."""
    if query in st.session_state.cache:
        return st.session_state.cache[query]

    api_key = random.choice(API_KEYS)  # Randomly select a key each time
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"query": query, "search_depth": "advanced"}
    
    try:
        response = requests.post(TAVILY_URL, json=payload, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        result = response.json()
        st.session_state.cache[query] = result
        return result
    except requests.exceptions.RequestException as e:
        return {"error": f"Tavily API request failed: {e}"}


# Function to create integrated summary with Gemini and multiple fallbacks
def summarize_with_gemini(snippets, variant_name):
    """
    Summarizes search snippets using the Gemini API with a fallback model strategy.
    
    CORRECTION: Updated the model list to valid and current model names.
    The previous list contained invalid names like "gemini-2.0-flash".
    """
    if not snippets:
        return f"No relevant snippets found for {variant_name} to summarize."
    
    prompt = f"Summarize the following search results about the genetic variant '{variant_name}' into a concise clinical interpretation. Focus on its clinical significance, pathogenicity classification (mentioning ACMG guidelines if available), associated conditions, and cite the evidence sources from the text.\n\n---BEGIN SNIPPETS---\n{' '.join(snippets)}\n---END SNIPPETS---"
    
    # List of valid models to try, from fastest/cheapest to most powerful
    models_to_try = [
        "gemini-1.5-flash-latest",
        "gemini-pro"
    ]

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            # Check if the response has text content
            if response.text:
                return response.text
            # Check for safety blocks or other reasons for empty response
            elif response.prompt_feedback.block_reason:
                 st.warning(f"Model {model_name} blocked the prompt. Reason: {response.prompt_feedback.block_reason.name}")
                 continue # Try the next model
            else:
                st.warning(f"Model {model_name} returned an empty response.")

        except Exception as e:
            st.warning(f"Model {model_name} failed with an error: {e}. Trying next model...")
    
    return f"Unable to generate summary for {variant_name} after trying all available models."

# --- Main Logic ---
# Search all variants and display integrated summaries
if st.button("Search Clinical Significance"):
    # Filter out empty variant inputs before processing
    active_variants = [v for v in st.session_state.variants if v.strip()]
    
    if not active_variants:
        st.warning("Please enter at least one variant to search.")
    else:
        with st.spinner("Searching for clinical significance and generating summaries..."):
            for variant in active_variants:
                query = f"clinical significance of genetic variant {variant}"
                st.write(f"**Searching for:** `{variant}`")
                
                result = search_tavily(query)
                
                # --- DEBUGGING LINE ---
                # Display the raw JSON result from the Tavily API to inspect its contents.
                st.json(result)

                if "error" in result:
                    st.error(result["error"])
                elif not result.get("results"):
                     st.warning(f"No search results found for `{variant}`.")
                else:
                    # Extract snippets, ensuring they are not empty
                    snippets = [r.get("snippet", "") for r in result.get("results", []) if r.get("snippet")]
                    summary = summarize_with_gemini(snippets, variant)
                    
                    st.markdown(f"### Gemini Clinical Summary for `{variant}`")
                    st.markdown(summary)
                
                st.write("---")

