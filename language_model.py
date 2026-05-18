def build_unigram(tokens):
    """Return unigrams (single tokens)."""
    return tokens

def build_bigram(tokens):
    """Return bigrams as list of strings."""
    if len(tokens) < 2:
        return []
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]

def build_trigram(tokens):
    """Return trigrams as list of strings."""
    if len(tokens) < 3:
        return []
    return [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(len(tokens) - 2)]

def get_ngrams(tokens, ngram_type='unigram'):
    """
    Dynamically select and build n-grams based on the token list.
    Supported ngram_type: 'unigram', 'bigram', 'trigram'
    """
    if ngram_type == 'unigram':
        return build_unigram(tokens)
    elif ngram_type == 'bigram':
        return build_bigram(tokens)
    elif ngram_type == 'trigram':
        return build_trigram(tokens)
    else:
        raise ValueError(f"Unsupported ngram_type: {ngram_type}")

def build_vocabulary(all_docs):
    """
    Build a unique vocabulary set from all documents.
    all_docs: A list of documents, where each document is a list of tokens/n-grams.
    """
    vocabulary = set()
    for doc_tokens in all_docs:
        vocabulary.update(doc_tokens)
    return vocabulary

def calculate_term_frequency(tokens):
    """
    Calculate the term frequency (TF) for a given list of tokens/n-grams.
    Returns a dictionary mapping token to its frequency.
    """
    freq_dict = {}
    for token in tokens:
        freq_dict[token] = freq_dict.get(token, 0) + 1
    return freq_dict

# Keep existing get_vectorizer so app.py doesn't crash during refactoring
from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer():
    return TfidfVectorizer(stop_words='english')
