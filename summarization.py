import re
import os
from multiprocessing import Pool
from preprocessing import preprocess

def summarize_text(text, num_sentences=2):
    """
    Generate an extractive summary of the text using word frequency-based scoring.
    """
    if not text or not text.strip():
        return ""
        
    # Split text into sentences using simple regex
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= num_sentences:
        return text.strip()
        
    # Preprocess document to get word frequencies
    words = preprocess(text)
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
        
    # Score sentences
    sentence_scores = []
    for idx, sentence in enumerate(sentences):
        sentence_words = preprocess(sentence)
        if not sentence_words:
            continue
        # Score is sum of word frequencies in the sentence
        score = sum(word_freq.get(w, 0) for w in sentence_words)
        # Length normalization to avoid bias towards long sentences
        score = score / (len(sentence_words) ** 0.5)
        sentence_scores.append((score, idx, sentence))
        
    if not sentence_scores:
        return " ".join(sentences[:num_sentences])
        
    # Get top sentences
    top_sentences = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:num_sentences]
    # Sort them back to original chronological index to preserve reading flow
    top_sentences_sorted = sorted(top_sentences, key=lambda x: x[1])
    
    summary = " ".join(s[2] for s in top_sentences_sorted)
    return summary

def _summarize_single(item):
    """Worker function for parallel summarization."""
    doc_id, info, num_sentences = item
    text = info["content"] if isinstance(info, dict) else info
    return doc_id, summarize_text(text, num_sentences)

def parallel_summarize(docs_dict, num_sentences=2, num_workers=None):
    """
    Summarize a collection of documents in parallel.
    docs_dict: Dictionary of {doc_id: info_dict_or_string}
    Returns: Dictionary of {doc_id: summary_string}
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 1
        
    items = [(doc_id, info, num_sentences) for doc_id, info in docs_dict.items()]
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(_summarize_single, items)
        
    return dict(results)
