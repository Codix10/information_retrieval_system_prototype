import os

def load_queries(filepath):
    """Load queries from a text file."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]
        
def ensure_directories(base_dir):
    """Ensure that the required directories exist."""
    dirs = [
        os.path.join(base_dir, 'data', 'nlp'),
        os.path.join(base_dir, 'data', 'cybersecurity'),
        os.path.join(base_dir, 'data', 'global_warming'),
        os.path.join(base_dir, 'queries')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def load_all_documents(data_dir):
    """
    Recursively load all .txt files from the data folder.
    Assigns unique document IDs and returns a dictionary of document metadata.
    """
    documents = {}
    doc_counter = 1
    
    if not os.path.exists(data_dir):
        return documents
        
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.txt'):
                filepath = os.path.join(root, file)
                # Assuming the immediate parent folder name is the domain
                domain = os.path.basename(root)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    continue
                    
                doc_id = f"doc{doc_counter}"
                documents[doc_id] = {
                    "domain": domain,
                    "filename": file,
                    "content": content
                }
                doc_counter += 1
                
    return documents
