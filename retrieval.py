import math
from preprocessing import preprocess

def process_query(query):
    """
    Preprocess the raw query text into a list of cleaned tokens.
    """
    return preprocess(query)

def score_tfidf(query_tokens, inverted_index, preprocessed_docs):
    """Calculate TF-IDF scores for documents matching the query."""
    N = len(preprocessed_docs)
    scores = {}
    
    for token in query_tokens:
        if token in inverted_index:
            doc_ids = inverted_index[token]
            df = len(doc_ids)
            # Standard IDF with +1 smoothing to avoid division by zero
            idf = math.log((N + 1) / (df + 1)) + 1
            
            for doc_id in doc_ids:
                doc_tokens = preprocessed_docs[doc_id]
                tf = doc_tokens.count(token)
                
                tfidf_score = tf * idf
                scores[doc_id] = scores.get(doc_id, 0.0) + tfidf_score
                
    return scores

def score_bm25(query_tokens, inverted_index, preprocessed_docs, k1=1.5, b=0.75):
    """Calculate Okapi BM25 scores for documents matching the query."""
    N = len(preprocessed_docs)
    if N == 0:
        return {}
        
    avgdl = sum(len(tokens) for tokens in preprocessed_docs.values()) / N
    scores = {}
    
    for token in query_tokens:
        if token in inverted_index:
            doc_ids = inverted_index[token]
            df = len(doc_ids)
            
            # BM25 IDF formula
            idf = math.log(((N - df + 0.5) / (df + 0.5)) + 1.0)
            
            for doc_id in doc_ids:
                doc_tokens = preprocessed_docs[doc_id]
                tf = doc_tokens.count(token)
                dl = len(doc_tokens)
                
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
