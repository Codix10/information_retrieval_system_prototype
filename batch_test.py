import os
from utils import load_all_documents, load_queries
from preprocessing import preprocess
from indexing import build_inverted_index
from retrieval import process_query, retrieve_documents, rank_documents

def run_batch_test(top_k=3, method='bm25'):
    """
    Runs batch query testing across all queries.
    Saves the output to results.txt in the requested format.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    queries_dir = os.path.join(base_dir, "queries")
    results_file = os.path.join(base_dir, "results.txt")
    
    print("Loading documents...")
    docs_dict = load_all_documents(data_dir)
    
    if not docs_dict:
        print("Warning: No documents found in data/. The index will be empty.")
    
    print("Preprocessing documents...")
    preprocessed_docs = {}
    for doc_id, info in docs_dict.items():
        preprocessed_docs[doc_id] = preprocess(info["content"])
        
    print("Building inverted index...")
    inverted_index = build_inverted_index(preprocessed_docs)
    
    # Load all queries from the queries folder
    query_files = ["nlp_queries.txt", "cyber_queries.txt", "gw_queries.txt"]
    all_queries = []
    
    for qf in query_files:
        q_path = os.path.join(queries_dir, qf)
        all_queries.extend(load_queries(q_path))
            
    print(f"Loaded {len(all_queries)} queries. Processing and ranking...")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        for query in all_queries:
            f.write(f"Query: {query}\n")
            f.write("Top Results:\n")
            
            # 1. Preprocess
            query_tokens = process_query(query)
            
            # 2. Retrieve
            results = retrieve_documents(query_tokens, inverted_index, preprocessed_docs, method=method)
            
            # 3. Rank
            ranked = rank_documents(results)
            
            # Output top_k docs
            for doc_id, _ in ranked[:top_k]:
                f.write(f"{doc_id}\n")
            f.write("\n")
            
    print(f"Results successfully saved to {results_file}")

if __name__ == "__main__":
    # You can change method to 'tfidf' or 'match_count'
    run_batch_test(top_k=3, method='bm25')
