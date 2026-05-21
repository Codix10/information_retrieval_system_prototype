import math
from preprocessing import preprocess

def process_query(query):
    """
    Preprocess the raw query text into a list of cleaned tokens.
    """
    return preprocess(query)

# Cache to avoid recomputing document lengths and average document length across queries
_stats_cache = {
    'preprocessed_docs_id': None,
    'doc_lengths': {},
    'avgdl': 0.0
}

def get_doc_stats(preprocessed_docs):
    global _stats_cache
    current_id = id(preprocessed_docs)
    if _stats_cache['preprocessed_docs_id'] == current_id:
        return _stats_cache['doc_lengths'], _stats_cache['avgdl']
    
    # Precompute statistics
    doc_lengths = {doc_id: len(tokens) for doc_id, tokens in preprocessed_docs.items()}
    N = len(doc_lengths)
    avgdl = sum(doc_lengths.values()) / N if N > 0 else 0.0
    
    _stats_cache = {
        'preprocessed_docs_id': current_id,
        'doc_lengths': doc_lengths,
        'avgdl': avgdl
    }
    return doc_lengths, avgdl

def score_tfidf(query_tokens, inverted_index, preprocessed_docs):
    """Calculate TF-IDF scores for documents matching the query."""
    N = len(preprocessed_docs)
    if N == 0:
        return {}
    scores = {}
    
    for token in query_tokens:
        if token in inverted_index:
            doc_tfs = inverted_index[token]
            df = len(doc_tfs)
            # Standard IDF with +1 smoothing to avoid division by zero
            idf = math.log((N + 1) / (df + 1)) + 1
            
            for doc_id, tf in doc_tfs.items():
                tfidf_score = tf * idf
                scores[doc_id] = scores.get(doc_id, 0.0) + tfidf_score
                
    return scores

def score_bm25(query_tokens, inverted_index, preprocessed_docs, k1=1.5, b=0.75):
    """Calculate Okapi BM25 scores for documents matching the query."""
    N = len(preprocessed_docs)
    if N == 0:
        return {}
        
    doc_lengths, avgdl = get_doc_stats(preprocessed_docs)
    scores = {}
    
    for token in query_tokens:
        if token in inverted_index:
            doc_tfs = inverted_index[token]
            df = len(doc_tfs)
            
            # BM25 IDF formula
            idf = math.log(((N - df + 0.5) / (df + 0.5)) + 1.0)
            
            for doc_id, tf in doc_tfs.items():
                dl = doc_lengths.get(doc_id, 0)
                
                # BM25 TF normalization
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (dl / avgdl))
                bm25_score = idf * (numerator / denominator)
                
                scores[doc_id] = scores.get(doc_id, 0.0) + bm25_score
                
    return scores

def retrieve_documents(query_tokens, inverted_index, preprocessed_docs=None, method='tfidf'):
    """
    Retrieve and score documents based on the selected ranking method.
    Supported methods: 'tfidf', 'bm25', 'match_count'
    """
    if method == 'tfidf' and preprocessed_docs is not None:
        return score_tfidf(query_tokens, inverted_index, preprocessed_docs)
    elif method == 'bm25' and preprocessed_docs is not None:
        return score_bm25(query_tokens, inverted_index, preprocessed_docs)
    else:
        # Fallback to simple term frequency match count if no docs provided or requested
        results = {}
        for token in query_tokens:
            if token in inverted_index:
                for doc_id in inverted_index[token]:
                    results[doc_id] = results.get(doc_id, 0) + 1
        return results

def rank_documents(results):
    """
    Rank documents based on their scores.
    
    results: Dictionary of {doc_id: score}
    Returns: A list of tuples sorted by score descending: [(doc_id, score), ...]
    """
    ranked = sorted(results.items(), key=lambda item: item[1], reverse=True)
    return ranked

# Keep existing retrieve function stub so app.py doesn't crash during refactoring
def retrieve(query, vectorizer, tfidf_matrix, documents, metadata, top_k=5):
    """Old retrieve function stub."""
    return []
