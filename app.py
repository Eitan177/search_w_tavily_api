import streamlit as st
import requests
import random
import google.generativeai as genai
import concurrent.futures

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
if "variants" not in st.session_state:
    st.session_state.variants = [""]
if "cache" not in st.session_state:
    st.session_state.cache = {}

# --- UI Components ---
def add_variant():
    st.session_state.variants.append("")

for i, variant in enumerate(st.session_state.variants):
    st.session_state.variants[i] = st.text_input(f"Variant {i+1}", value=variant, key=f"variant_{i}")

if st.button("Add another variant"):
    add_variant()

# --- API Functions ---
def search_tavily(query):
    """Performs a search using the Tavily API and caches the result."""
    if query in st.session_state.cache:
        return st.session_state.cache[query]
    api_key = random.choice(API_KEYS)
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"query": query, "search_depth": "advanced"}
    try:
        response = requests.post(TAVILY_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        st.session_state.cache[query] = result
        return result
    except requests.exceptions.RequestException as e:
        return {"error": f"Tavily API request failed: {e}"}

def summarize_with_gemini(search_content, variant_name):
    """
    Summarizes search content using the Gemini API.
    
    FIX: This function no longer calls st.warning. It returns warnings as data
    to be displayed by the main thread, avoiding Streamlit's threading issues.
    """
    warnings = []
    if not search_content:
        summary = f"No relevant content found for {variant_name} to summarize."
        return {"summary": summary, "warnings": warnings}
        
    prompt = f"Summarize the following search results about the genetic variant '{variant_name}' into a concise clinical interpretation. Focus on its clinical significance, pathogenicity classification (mentioning ACMG guidelines if available), associated conditions, and cite the evidence sources from the text.\n\n---BEGIN SEARCH CONTENT---\n{' '.join(search_content)}\n---END SEARCH CONTENT---"
    models_to_try = ["gemini-1.5-flash-latest", "gemini-pro"]

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response.text:
                # Success, return the summary and any warnings from previous failed models
                return {"summary": response.text, "warnings": warnings}
            elif response.prompt_feedback.block_reason:
                warnings.append(f"Model {model_name} blocked the prompt. Reason: {response.prompt_feedback.block_reason.name}")
                continue
            else:
                warnings.append(f"Model {model_name} returned an empty response.")
        except Exception as e:
            warnings.append(f"Model {model_name} failed with an error: {e}. Trying next model...")
            
    final_summary = f"Unable to generate summary for {variant_name} after trying all available models."
    return {"summary": final_summary, "warnings": warnings}

# --- Main Logic ---
def process_variant(variant):
    """
    Encapsulates the full search and summary process for a single variant.
    This function is designed to be run in a separate thread.
    """
    query = f"clinical significance of genetic variant {variant}"
    result = search_tavily(query)
    
    if "error" in result:
        summary_data = {"summary": f"Error searching for variant: {result['error']}", "warnings": []}
    elif not result.get("results"):
        summary_data = {"summary": f"No search results found for `{variant}`.", "warnings": []}
    else:
        content_list = [r.get("content", "") for r in result.get("results", []) if r.get("content")]
        summary_data = summarize_with_gemini(content_list, variant)
        
    return {"variant": variant, "summary_data": summary_data}

if st.button("Search Clinical Significance"):
    active_variants = [v for v in st.session_state.variants if v.strip()]
    
    if not active_variants:
        st.warning("Please enter at least one variant to search.")
    else:
        with st.spinner(f"Searching for {len(active_variants)} variant(s) and generating summaries... This may take a moment."):
            # Use a ThreadPoolExecutor to run API calls concurrently
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # map the process_variant function to all active_variants
                results = list(executor.map(process_variant, active_variants))

            # Display results after all threads have completed
            for res in results:
                st.markdown(f"### Gemini Clinical Summary for `{res['variant']}`")
                
                # FIX: Display warnings from the main thread
                if res["summary_data"]["warnings"]:
                    for warning_text in res["summary_data"]["warnings"]:
                        st.warning(warning_text)

                st.markdown(res["summary_data"]["summary"])
                st.write("---")
