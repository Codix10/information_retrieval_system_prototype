def precision(retrieved_docs, relevant_docs):
    """
    Calculate Precision: (True Positives) / (Total Retrieved Documents)
    Returns result rounded to 3 decimal places.
    """
    if not retrieved_docs:
        return 0.0
        
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    
    true_positives = len(retrieved_set.intersection(relevant_set))
    p = true_positives / len(retrieved_set)
    return round(p, 3)

def recall(retrieved_docs, relevant_docs):
    """
    Calculate Recall: (True Positives) / (Total Relevant Documents)
    Returns result rounded to 3 decimal places.
    """
    if not relevant_docs:
        return 0.0
        
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    
    true_positives = len(retrieved_set.intersection(relevant_set))
    r = true_positives / len(relevant_set)
    return round(r, 3)

def f1_score(precision_val, recall_val):
    """
    Calculate F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    Returns result rounded to 3 decimal places.
    """
    if (precision_val + recall_val) == 0:
        return 0.0
        
    f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val)
    return round(f1, 3)
