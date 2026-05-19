import streamlit as st
import os
from utils import load_all_documents, ensure_directories
from preprocessing import preprocess
from indexing import build_inverted_index
from retrieval import process_query, retrieve_documents, rank_documents
from language_model import build_vocabulary
from multiprocessing import current_process

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ensure_directories(BASE_DIR)

# Load and index data efficiently
@st.cache_resource
def initialize_system():
    data_dir = os.path.join(BASE_DIR, "data")
    docs_dict = load_all_documents(data_dir)
    
    if not docs_dict:
        return {}, {}, {}, 0, 0
        
    from preprocessing import parallel_preprocess
    preprocessed_docs = parallel_preprocess(docs_dict)
        
    inverted_index = build_inverted_index(preprocessed_docs)
    
    vocab = build_vocabulary(preprocessed_docs.values())
    vocab_size = len(vocab)
    
    domains = set(info["domain"] for info in docs_dict.values())
    num_domains = 3
    
    return docs_dict, preprocessed_docs, inverted_index, vocab_size, num_domains

# Only execute page configuration, indexing, and Streamlit widgets in the MainProcess
if current_process().name == 'MainProcess':
    st.set_page_config(page_title="IR System", layout="wide")

    with st.spinner("Loading corpus and building index..."):
        docs_dict, preprocessed_docs, inverted_index, vocab_size, num_domains = initialize_system()

    # Sidebar configuration
    st.sidebar.header("Project Information")
    if docs_dict:
        st.sidebar.metric("Total Documents", len(docs_dict))
        st.sidebar.metric("Total Vocabulary Size", vocab_size)
        st.sidebar.metric("Indexed Domains", num_domains)
    else:
        st.sidebar.warning("No documents found in the data/ directory.")

    st.sidebar.divider()

    st.sidebar.header("Recent Searches")
    if "recent_searches" not in st.session_state:
        st.session_state.recent_searches = []

    if st.session_state.recent_searches:
        for s in reversed(st.session_state.recent_searches):
            st.sidebar.text(f"🔍 {s}")
    else:
        st.sidebar.caption("No recent searches.")

    # Main Interface Header
    st.divider()
    st.title("Multi-Domain Information Retrieval System")
    st.markdown(f"Search across {num_domains} research domains" if num_domains > 0 else "Search across research domains")

    # Initialize query in session state so buttons can update it
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    def set_query(q):
        st.session_state.search_query = q

    # Search Box & Button on same line
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input("Search", value=st.session_state.search_query, label_visibility="collapsed", placeholder="Enter your search query here...")
    with col2:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    # Suggested Queries
    st.write("Suggested Queries:")
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        if st.button("ransomware attacks", use_container_width=True):
            query = "ransomware attacks"
            search_clicked = True
    with s_col2:
        if st.button("transformer models", use_container_width=True):
            query = "transformer models"
            search_clicked = True
    with s_col3:
        if st.button("sea level rise", use_container_width=True):
            query = "sea level rise"
            search_clicked = True

    st.divider()

    # Results Section
    if search_clicked:
        if not query:
            st.warning("Please enter a query.")
        elif not docs_dict:
            st.error("System is not initialized because there are no documents.")
        else:
            # Update recent searches
            if query not in st.session_state.recent_searches:
                st.session_state.recent_searches.append(query)
                if len(st.session_state.recent_searches) > 10:
                    st.session_state.recent_searches.pop(0)
                    
            with st.spinner("Searching and ranking..."):
                query_tokens = process_query(query)
                results = retrieve_documents(query_tokens, inverted_index, preprocessed_docs, method='bm25')
                ranked = rank_documents(results)
                
                # Take top 5 documents
                top_k = 5
                top_results = ranked[:top_k]
                
                st.subheader("Results:")
                if top_results:
                    for i, (doc_id, score) in enumerate(top_results):
                        doc_info = docs_dict[doc_id]
                        display_domain = doc_info['domain'].replace('_', ' ').title()
                        content = doc_info['content']
                        
                        # Result Card
                        st.markdown("---")
                        st.markdown(f"### Result {i+1}")
                        st.write(f"**Doc ID:** {doc_id}")
                        st.write(f"**Filename:** {doc_info['filename']}")
                        st.write(f"**Domain:** {display_domain}")
                        st.write(f"**Score:** {score:.4f}")
                        
                        # Show summary by default
                        from summarization import summarize_text
                        summary = summarize_text(content, num_sentences=2)
                        st.write(f"**Summary:** {summary}")
                        
                        # Optional: View full document in expander
                        with st.expander("View Full Document"):
                            st.write(content)
                    st.markdown("---")
                else:
                    st.info("No matching documents found.")
