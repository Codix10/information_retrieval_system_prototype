def build_inverted_index(preprocessed_docs):
    """
    Build an inverted index from documents.
    
    preprocessed_docs: A dictionary mapping doc_id to a list of preprocessed tokens/n-grams.
                       Example: {"doc1": ["security", "climate"], "doc2": ["security"]}
    
    Returns:
    An inverted index dictionary mapping each token to a list of unique doc_ids.
    Example: {"security": ["doc1", "doc2"], "climate": ["doc1"]}
    """
    inverted_index = {}
    
    for doc_id, tokens in preprocessed_docs.items():
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = {}
            inverted_index[token][doc_id] = inverted_index[token].get(doc_id, 0) + 1
            
    return inverted_index

# Keep existing build_index so app.py doesn't crash during refactoring
def build_index(documents, vectorizer):
    """Build the TF-IDF index for the given documents using scikit-learn."""
    if not documents:
        return None
    return vectorizer.fit_transform(documents)
