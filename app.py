import streamlit as st
import requests
import random
import google.generativeai as genai
import concurrent.futures
import json
from urllib.parse import quote

# --- Page and API Configuration ---
st.set_page_config(
    page_title="Variant Intelligence Aggregator",
    layout="wide",
)

# --- Secret/Key Management ---
# Configure Gemini API key
try:
    GEMINI_API_KEY = st.secrets["GEMINI_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("Gemini API key not found. Please set `GEMINI_KEY` in your Streamlit secrets.")
    st.stop()

# Store Tavily API keys
try:
    TAVILY_API_KEYS = [
        st.secrets["TAVILY_KEY_1"], st.secrets["TAVILY_KEY_2"], st.secrets["TAVILY_KEY_3"],
        st.secrets["TAVILY_KEY_4"], st.secrets["TAVILY_KEY_5"], st.secrets["TAVILY_KEY_6"]
    ]
except (KeyError, FileNotFoundError):
    st.error("Tavily API keys not found. Please set `TAVILY_KEY_1` through `TAVILY_KEY_6` in your Streamlit secrets.")
    st.stop()

# Get OncoKB API Token
ONCOKB_API_TOKEN = st.secrets.get("ONCOKB_API_KEY")


# --- Constants ---
TAVILY_URL = "https://api.tavily.com/search"
ONCOKB_URL = "https://www.oncokb.org/api/v1"
GEMINI_MODELS_API = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"


# --- Session State Initialization ---
if "variants" not in st.session_state:
    st.session_state.variants = [""]
if "cache" not in st.session_state:
    st.session_state.cache = {}
if "query_template" not in st.session_state:
    st.session_state.query_template = "clinical significance of genetic variant {variant}"
if "tumor_type" not in st.session_state:
    st.session_state.tumor_type = ""
if "include_tavily" not in st.session_state:
    st.session_state.include_tavily = True
if "include_oncokb" not in st.session_state:
    st.session_state.include_oncokb = True
if "include_perplexity" not in st.session_state:
    st.session_state.include_perplexity = True
if "available_models" not in st.session_state:
    st.session_state.available_models = None


# --- UI Components ---
st.title("Variant Intelligence Aggregator")
st.info("This tool aggregates information from web searches (Tavily) and the OncoKB database.")

with st.sidebar:
    st.header("Search Configuration")
    st.session_state.query_template = st.text_area(
        "Web Search Query Template",
        value=st.session_state.query_template,
        help="Define the base web search query. Use `{variant}` as a placeholder."
    )
    st.session_state.tumor_type = st.text_input(
        "Tumor Type (Optional)",
        value=st.session_state.tumor_type,
        help="Applies to both Web and OncoKB searches (e.g., 'Melanoma')."
    )
    
    st.header("Data Sources to Query")
    st.session_state.include_tavily = st.checkbox("Tavily Web Search", value=st.session_state.include_tavily)
    st.session_state.include_oncokb = st.checkbox("OncoKB Database", value=st.session_state.include_oncokb)
    st.session_state.include_perplexity = st.checkbox("Perplexity Quick Links", value=st.session_state.include_perplexity)

    st.header("API Status")
    st.success("Gemini API Key: Found")
    st.success("Tavily API Keys: Found")
    if ONCOKB_API_TOKEN:
        st.success("OncoKB API Key: Found")
    else:
        st.warning("OncoKB API Key: Not Found. OncoKB search will be disabled.")


st.header("Variant Input")
def add_variant():
    st.session_state.variants.append("")

for i, variant in enumerate(st.session_state.variants):
    st.session_state.variants[i] = st.text_input(f"Variant {i+1}", value=variant, key=f"variant_{i}")

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Add another variant"):
        add_variant()
with col2:
    search_button_pressed = st.button("Search All Variants", type="primary")


# --- API Functions (Thread-Safe) ---
def fetch_from_tavily_headless(query):
    api_key = random.choice(TAVILY_API_KEYS)
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"query": query, "search_depth": "advanced"}
    try:
        response = requests.post(TAVILY_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Tavily API request failed: {e}"}

def fetch_from_oncokb_headless(hugo_symbol, alteration, tumor_type):
    if not ONCOKB_API_TOKEN:
        return {"error": "OncoKB API Token not configured."}
    
    hugo_symbol_upper = hugo_symbol.upper()
    alteration_upper = alteration.upper()
    tumor_type_upper = tumor_type.upper()

    api_alteration = alteration_upper[2:] if alteration_upper.startswith('P.') else alteration_upper
    api_url = f"{ONCOKB_URL}/annotate/mutations/byProteinChange?hugoSymbol={hugo_symbol_upper}&alteration={api_alteration}"
    
    if tumor_type:
        api_url += f"&tumorType={tumor_type_upper}"
    
    headers = {'Authorization': f'Bearer {ONCOKB_API_TOKEN}', 'Accept': 'application/json'}
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        try: return e.response.json()
        except ValueError: return {'error': f"API Error: Status {e.response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {'error': f"Network Error: {e}"}

@st.cache_data
def get_available_models():
    """Gets a list of available Gemini models that support generateContent."""
    try:
        response = requests.get(GEMINI_MODELS_API)
        response.raise_for_status()
        models_data = response.json()
        return [
            m['name'].split('/')[-1] for m in models_data.get('models', [])
            if 'generateContent' in m.get('supportedGenerationMethods', [])
        ]
    except Exception as e:
        # This function is cached, so st.error is safe here as it runs once on main thread
        st.error(f"Could not fetch available models: {e}")
        return []

def summarize_with_gemini(prompt, available_models):
    """
    Summarizes content using Gemini, with a dynamic fallback to available models.
    FIX: This function now receives the list of available models as an argument
    and does not access st.session_state, making it thread-safe.
    """
    warnings = []
    last_error = None
    
    preferred_models = ["gemini-1.5-flash-latest", "gemini-pro"]
    models_to_try = preferred_models

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response.text:
                return {"summary": response.text, "warnings": warnings, "error": None}
            warnings.append(f"Model '{model_name}' returned an empty or blocked response.")
            last_error = "Model returned an empty or blocked response."
        except Exception as e:
            last_error = str(e)
            warnings.append(f"Model '{model_name}' failed: {last_error}")
            if "404" in last_error and "is not found" in last_error:
                if available_models:
                    warnings.append(f"'{model_name}' not found. Trying other available models...")
                    dynamic_models = [m for m in available_models if m != model_name]
                    for dynamic_model_name in dynamic_models:
                        try:
                            model = genai.GenerativeModel(dynamic_model_name)
                            response = model.generate_content(prompt)
                            if response.text:
                                return {"summary": response.text, "warnings": warnings, "error": None}
                            warnings.append(f"Model '{dynamic_model_name}' returned an empty or blocked response.")
                            last_error = "Model returned an empty or blocked response."
                        except Exception as e_dynamic:
                            last_error = str(e_dynamic)
                            warnings.append(f"Model '{dynamic_model_name}' failed: {last_error}")
                    break

    final_summary = "Unable to generate summary."
    if last_error:
        final_summary += f" Last known error: {last_error}"
        
    return {"summary": final_summary, "warnings": warnings, "error": last_error}


# --- Processing Logic ---
def process_tavily_search(variant, search_result, available_models):
    sources = []
    if "error" in search_result:
        summary_data = {"summary": f"Error during search: {search_result['error']}", "warnings": []}
    else:
        content = [r.get("content", "") for r in search_result.get("results", [])]
        sources = [r.get("url") for r in search_result.get("results", []) if r.get("url")]
        prompt = f"Summarize the clinical significance of '{variant}' based on the following text:\n\n{' '.join(content)}"
        summary_data = summarize_with_gemini(prompt, available_models)
    return {"variant": variant, "summary_data": summary_data, "sources": sources}

def process_oncokb_search(variant_gene, variant_alt, search_result, available_models):
    prompt_text = ""
    if "error" in search_result:
        summary_data = {"summary": f"OncoKB API Error: {search_result['error']}", "warnings": []}
    elif search_result.get('query', {}).get('variant') == "UNKNOWN":
        summary_data = {"summary": "Variant not found in OncoKB.", "warnings": []}
    else:
        prompt_text = f"Please provide a clinical summary for the variant {variant_gene} {variant_alt} based on the following curated data from OncoKB:\n\n"
        
        gene_summary = search_result.get('geneSummary')
        if gene_summary:
            prompt_text += f"**Gene Summary:** {gene_summary}\n\n"

        variant_summary = search_result.get('variantSummary')
        if variant_summary:
            prompt_text += f"**Variant Summary:** {variant_summary}\n\n"
        
        diag_implications = search_result.get('diagnosticImplications')
        if diag_implications:
            prompt_text += "**Diagnostic Implications:**\n"
            for item in diag_implications:
                level = item.get('levelOfEvidence', 'N/A').replace('_', ' ')
                tumor_name = item.get('tumorType', {}).get('name', 'N/A')
                prompt_text += f"- **Tumor Type:** {tumor_name} (Level: {level})\n"
            prompt_text += "\n"

        prog_implications = search_result.get('prognosticImplications')
        if prog_implications:
            prompt_text += "**Prognostic Implications:**\n"
            for item in prog_implications:
                level = item.get('levelOfEvidence', 'N/A').replace('_', ' ')
                tumor_name = item.get('tumorType', {}).get('name', 'N/A')
                prompt_text += f"- **Tumor Type:** {tumor_name} (Level: {level})\n"
            prompt_text += "\n"

        treatments = search_result.get('treatments')
        if treatments:
            prompt_text += "**Therapeutic Implications:**\n"
            for treatment in treatments:
                drugs = ", ".join([d['drugName'] for d in treatment.get('drugs', [])])
                level = treatment.get('level', 'N/A').replace('_', ' ')
                indication = treatment.get('indication', {}).get('name', 'N/A')
                prompt_text += f"- **Drugs:** {drugs} for {indication} (Level: {level})\n"
        
        summary_data = summarize_with_gemini(prompt_text, available_models)

    return {"variant": f"{variant_gene} {variant_alt}", "summary_data": summary_data, "oncokb_data": search_result, "prompt": prompt_text}


# --- Main Application Flow ---
if search_button_pressed:
    active_variants = [v.strip() for v in st.session_state.variants if v.strip()]
    if not active_variants:
        st.warning("Please enter at least one variant to search.")
    else:
        tabs_to_show = []
        if st.session_state.include_tavily:
            tabs_to_show.append("Web Search (Tavily)")
        if st.session_state.include_oncokb:
            tabs_to_show.append("OncoKB Search")
        if st.session_state.include_perplexity:
            tabs_to_show.append("Quick Links (Perplexity)")

        if not tabs_to_show:
            st.warning("Please select at least one data source to query.")
        else:
            created_tabs = st.tabs(tabs_to_show)
            tab_map = {name: tab for name, tab in zip(tabs_to_show, created_tabs)}

            query_template = st.session_state.query_template
            tumor_type = st.session_state.tumor_type.strip()
            
            # --- FIX: Fetch available models on the main thread before starting workers ---
            if st.session_state.available_models is None:
                st.session_state.available_models = get_available_models()
            available_models_list = st.session_state.available_models
            
            with st.spinner("Fetching and summarizing results..."):

                if st.session_state.include_perplexity and "Quick Links (Perplexity)" in tab_map:
                    with tab_map["Quick Links (Perplexity)"]:
                        st.markdown("### Quick Search Links")
                        for variant in active_variants:
                            base_query = query_template.format(variant=variant)
                            full_query = f"{base_query} in {tumor_type}" if tumor_type else base_query
                            perplexity_url = f"https://www.perplexity.ai/search?q={quote(full_query)}"
                            st.markdown(f"**For `{variant}`:**")
                            st.link_button("Ask Perplexity.ai", perplexity_url)
                        st.write("---")

                if st.session_state.include_tavily and "Web Search (Tavily)" in tab_map:
                    with tab_map["Web Search (Tavily)"]:
                        st.markdown("### Web Search Summaries (Tavily + Gemini)")
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future_to_variant = {executor.submit(fetch_from_tavily_headless, f"{query_template.format(variant=v)} {'in ' + tumor_type if tumor_type else ''}"): v for v in active_variants}
                            search_results = {future_to_variant[future]: future.result() for future in concurrent.futures.as_completed(future_to_variant)}
                            
                            # --- FIX: Pass the list of models to the processing function ---
                            summary_futures = {executor.submit(process_tavily_search, v, search_results.get(v, {}), available_models_list): v for v in active_variants}
                            for future in concurrent.futures.as_completed(summary_futures):
                                res = future.result()
                                st.markdown(f"#### Summary for `{res['variant']}`")
                                st.markdown(res['summary_data']['summary'])
                                if res.get("sources"):
                                    with st.expander("Show Sources"):
                                        for url in res["sources"]: st.markdown(f"- {url}")
                                st.divider()

                if st.session_state.include_oncokb and "OncoKB Search" in tab_map:
                    with tab_map["OncoKB Search"]:
                        st.markdown("### OncoKB Summaries (OncoKB + Gemini)")
                        if not ONCOKB_API_TOKEN:
                            st.error("OncoKB search is disabled. Please add your `ONCOKB_API_KEY` to your Streamlit secrets.")
                        else:
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                parsed_variants = []
                                for v in active_variants:
                                    parts = v.split(' ', 1)
                                    if len(parts) == 2: parsed_variants.append({'gene': parts[0], 'alt': parts[1], 'original': v})
                                    else: st.warning(f"Could not parse gene/alteration for '{v}'. Skipping OncoKB search.")
                                
                                future_to_variant = {executor.submit(fetch_from_oncokb_headless, pv['gene'], pv['alt'], tumor_type): pv for pv in parsed_variants}
                                search_results = {future_to_variant[future]['original']: future.result() for future in concurrent.futures.as_completed(future_to_variant)}
                                
                                # --- FIX: Pass the list of models to the processing function ---
                                summary_futures = {executor.submit(process_oncokb_search, pv['gene'], pv['alt'], search_results.get(pv['original'], {}), available_models_list): pv['original'] for pv in parsed_variants}
                                for future in concurrent.futures.as_completed(summary_futures):
                                    res = future.result()
                                    st.markdown(f"#### Summary for `{res['variant']}`")
                                    st.markdown(res['summary_data']['summary'])
                                    
                                    if res.get('oncokb_data') and 'query' in res['oncokb_data']:
                                        q = res['oncokb_data']['query']
                                        link = f"https://www.oncokb.org/gene/{q.get('hugoSymbol', '')}/{q.get('alteration', '')}"
                                        st.link_button("View on OncoKB", link)
                                        with st.expander("Show Debugging Info"):
                                            st.markdown("**Prompt sent to Gemini:**")
                                            st.text(res.get("prompt", "No prompt was generated."))
                                            st.markdown("**Raw OncoKB Data:**")
                                            st.json(res['oncokb_data'])
                                    st.divider()

