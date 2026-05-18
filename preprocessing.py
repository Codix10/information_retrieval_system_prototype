import re
import string

CUSTOM_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "when", "where", "how", "who", "which", "this", "that", "these", "those",
    "then", "just", "so", "than", "such", "both", "through", "about", "for",
    "is", "of", "while", "during", "to", "in", "out", "on", "off", "over", "under",
    "again", "further", "once", "here", "there", "why", "all", "any", "each",
    "few", "more", "most", "other", "some", "no", "nor", "not", "only", "own",
    "same", "too", "very", "s", "t", "can", "will", "don", "should", "now",
    "it", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "whom", "am", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "until", "at", "by", "with", "against", "between", "into", "before", "after",
    "above", "below", "from", "up", "down"
}

def lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def tokenize(text):
    # Use re to split by whitespace and get alphanumeric tokens
    return re.findall(r'\S+', text)

def remove_stopwords(tokens):
    return [word for word in tokens if word not in CUSTOM_STOPWORDS]

def stem_word(word):
    # Basic suffix stripping
    if len(word) > 4 and word.endswith('ing'):
        return word[:-3]
    elif len(word) > 3 and word.endswith('ed'):
        return word[:-2]
    elif len(word) > 3 and word.endswith('ly'):
        return word[:-2]
    elif len(word) > 2 and word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word

def preprocess(text):
    if not text:
        return []
    
    text = lowercase(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = [stem_word(word) for word in tokens]
    
    return tokens
