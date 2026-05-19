import os
from multiprocessing import Pool
from utils import load_all_documents, load_queries
from preprocessing import preprocess, parallel_preprocess
from indexing import build_inverted_index
from retrieval import process_query, retrieve_documents, rank_documents

# Global variables for child processes to avoid massive IPC overhead
_worker_inverted_index = None
_worker_preprocessed_docs = None

def _init_worker(inverted_index, preprocessed_docs):
    """Initialize global variables in worker processes."""
    global _worker_inverted_index, _worker_preprocessed_docs
    _worker_inverted_index = inverted_index
    _worker_preprocessed_docs = preprocessed_docs

def _test_single_query(args):
    """Worker function to process, retrieve, and rank a single query."""
    query, method, top_k = args
    query_tokens = process_query(query)
    results = retrieve_documents(query_tokens, _worker_inverted_index, _worker_preprocessed_docs, method=method)
    ranked = rank_documents(results)
    top_docs = [doc_id for doc_id, _ in ranked[:top_k]]
    return query, top_docs

def parallel_batch_test(all_queries, inverted_index, preprocessed_docs, method, top_k, num_workers=None):
    """Run batch queries in parallel using multiprocessing Pool with global initialization."""
    if num_workers is None:
        num_workers = os.cpu_count() or 1
        
    tasks = [(query, method, top_k) for query in all_queries]
    
    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(inverted_index, preprocessed_docs)
    ) as pool:
        results = pool.map(_test_single_query, tasks)
        
    return results

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
    
    print("Preprocessing documents in parallel...")
    preprocessed_docs = parallel_preprocess(docs_dict)
        
    print("Building inverted index...")
    inverted_index = build_inverted_index(preprocessed_docs)
    
    # Load all queries from the queries folder
    query_files = ["nlp_queries.txt", "cyber_queries.txt", "gw_queries.txt"]
    all_queries = []
    
    for qf in query_files:
        q_path = os.path.join(queries_dir, qf)
        all_queries.extend(load_queries(q_path))
            
    print(f"Loaded {len(all_queries)} queries. Processing and ranking in parallel...")
    results = parallel_batch_test(all_queries, inverted_index, preprocessed_docs, method, top_k)
    
    print("Saving results...")
    with open(results_file, 'w', encoding='utf-8') as f:
        for query, top_docs in results:
            f.write(f"Query: {query}\n")
            f.write("Top Results:\n")
            for doc_id in top_docs:
                f.write(f"{doc_id}\n")
            f.write("\n")
            
    print(f"Results successfully saved to {results_file}")

if __name__ == "__main__":
    # You can change method to 'tfidf' or 'match_count'
    run_batch_test(top_k=3, method='bm25')

