import streamlit as st
import requests
import random
import google.generativeai as genai
import concurrent.futures
from urllib.parse import quote

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
if "query_template" not in st.session_state:
    st.session_state.query_template = "clinical significance of genetic variant {variant}"
# --- CHANGE ---: Add tumor_type to session state
if "tumor_type" not in st.session_state:
    st.session_state.tumor_type = ""


# --- UI Components ---
st.session_state.query_template = st.text_area(
    "Search Query Template",
    value=st.session_state.query_template,
    help="Define the base search query. Use `{variant}` as a placeholder for the variant name. The tumor type will be appended if provided."
)

# --- CHANGE ---: Add a text input for the optional tumor type
st.session_state.tumor_type = st.text_input(
    "Tumor Type (Optional)",
    value=st.session_state.tumor_type,
    help="If provided, this will be added to the search query (e.g., '...in breast cancer')."
)


def add_variant():
    st.session_state.variants.append("")

for i, variant in enumerate(st.session_state.variants):
    st.session_state.variants[i] = st.text_input(f"Variant {i+1}", value=variant, key=f"variant_{i}")

if st.button("Add another variant"):
    add_variant()

# --- API Functions ---
def fetch_from_tavily_headless(query):
    """
    Performs a search using the Tavily API. 
    This version is 'headless' and does NOT interact with st.session_state,
    making it safe to run in a background thread.
    """
    api_key = random.choice(API_KEYS)
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"query": query, "search_depth": "advanced"}
    try:
        response = requests.post(TAVILY_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Tavily API request failed: {e}"}

def summarize_with_gemini(search_content, variant_name):
    """
    Summarizes search content using the Gemini API.
    Returns warnings as data to be displayed by the main thread.
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
def process_variant_summary(variant, search_result):
    """
    Processes a single variant's search result to generate a summary.
    This function is designed to be run in a separate thread.
    """
    sources = []
    if "error" in search_result:
        summary_data = {"summary": f"Error during search: {search_result['error']}", "warnings": []}
    elif not search_result.get("results"):
        summary_data = {"summary": f"No search results found for `{variant}`.", "warnings": []}
    else:
        content_list = [r.get("content", "") for r in search_result.get("results", []) if r.get("content")]
        sources = [r.get("url") for r in search_result.get("results", []) if r.get("url")]
        summary_data = summarize_with_gemini(content_list, variant)
        
    return {"variant": variant, "summary_data": summary_data, "sources": sources}

if st.button("Search Clinical Significance"):
    active_variants = [v for v in st.session_state.variants if v.strip()]
    
    if not active_variants:
        st.warning("Please enter at least one variant to search.")
    else:
        query_template = st.session_state.query_template
        # --- CHANGE ---: Get tumor type from session state
        tumor_type = st.session_state.tumor_type.strip()

        st.markdown("### Quick Search Links")
        for variant in active_variants:
            # --- CHANGE ---: Construct the full query including the optional tumor type
            base_query = query_template.format(variant=variant)
            full_query = f"{base_query} in {tumor_type}" if tumor_type else base_query
            
            perplexity_url = f"https://www.perplexity.ai/search?q={quote(full_query)}"
            st.markdown(f"**For `{variant}`:**")
            st.link_button("Ask Perplexity.ai", perplexity_url)
        st.write("---")
        
        with st.spinner(f"Now fetching and summarizing results for {len(active_variants)} variant(s)... This may take a moment."):
            
            variants_to_fetch = []
            variant_to_search_result = {}
            # --- CHANGE ---: This dictionary will hold the final query for each variant
            variant_queries = {}

            for variant in active_variants:
                base_query = query_template.format(variant=variant)
                full_query = f"{base_query} in {tumor_type}" if tumor_type else base_query
                variant_queries[variant] = full_query
                
                if full_query in st.session_state.cache:
                    variant_to_search_result[variant] = st.session_state.cache[full_query]
                else:
                    variants_to_fetch.append(variant)

            if variants_to_fetch:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_variant = {
                        executor.submit(fetch_from_tavily_headless, variant_queries[variant]): variant 
                        for variant in variants_to_fetch
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_variant):
                        variant = future_to_variant[future]
                        try:
                            search_result = future.result()
                            variant_to_search_result[variant] = search_result
                            query = variant_queries[variant]
                            st.session_state.cache[query] = search_result
                        except Exception as exc:
                            variant_to_search_result[variant] = {"error": f"Generated an exception during search: {exc}"}

            final_results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                summary_futures = {
                    executor.submit(process_variant_summary, variant, result): variant
                    for variant, result in variant_to_search_result.items()
                }
                for future in concurrent.futures.as_completed(summary_futures):
                    final_results.append(future.result())

            st.markdown("### Detailed Summaries")
            results_dict = {res['variant']: res for res in final_results}
            for variant in active_variants:
                res = results_dict.get(variant)
                if res:
                    st.markdown(f"#### Gemini Clinical Summary for `{res['variant']}`")
                    # --- CHANGE ---: Display the final, full query that was sent
                    st.info(f"**Query sent:** `{variant_queries[variant]}`")

                    if res["summary_data"]["warnings"]:
                        for warning_text in res["summary_data"]["warnings"]:
                            st.warning(warning_text)
                    st.markdown(res["summary_data"]["summary"])
                    
                    if res.get("sources"):
                        st.markdown("**Sources:**")
                        for source_url in res["sources"]:
                            st.markdown(f"- [{source_url}]({source_url})")

                    st.write("---")

