# Information Retrieval System (IR) Prototype
A small-scale Information Retrieval system that retrieves documents from a corpus across three domains: Natural Language Processing, Cyber Security, and Global Warming.

## Features
- **Boolean Retrieval**: Supports basic boolean operations (`AND`, `OR`, `NOT`) for precise query formulation.
- **Vector Space Model**: Implements TF-IDF weighting and cosine similarity for ranked retrieval.
- **Domain-Specific Retrieval**: Optimized for IR concepts, security threats, and climate science.

## Setup

### Prerequisites
- Python 3.6+
- Required packages: `nltk`, `pandas`, `matplotlib`, `scikit-learn`

### Installation
1. Clone the repository (if available).
2. Install dependencies:
   ```bash
   pip install nltk pandas matplotlib scikit-learn
   ```
3. Download NLTK data:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

## Usage

### 1. Boolean Retrieval
Use the `boolean_retrieval` function to perform logical queries on the document collection.

**Example:**
```python
from core.boolean_model import boolean_search, load_corpus

# Load the corpus
corpus = load_corpus("path/to/documents")

# Perform a query
results = boolean_search("Natural Language Processing AND deep learning", corpus)
print(results)
```

### 2. Vector Space Model
Use the `vector_space_retrieval` function for ranked search results based on relevance.

**Example:**
```python
from core.vector_space_model import vector_space_retrieval, build_index

# Build the inverted index and document vectors
index = build_index("path/to/documents")

# Search for the most relevant documents
query = "How do neural networks process text?"
results = vector_space_retrieval(query, index, top_k=10)
print(results)
```

## System Components
- **Data**: Sample document collections for each domain stored in the `data/` directory.
- **Core**: Implements the retrieval algorithms in `core/`.
  - `boolean_model.py`: Boolean logic operations.
  - `vector_space_model.py`: TF-IDF and cosine similarity implementation.
- **Visualization**: Generates precision-recall curves and performance metrics in `visualization/`.
- **Scripts**: Utility scripts for data processing and testing.

## Testing
Run the test scripts to validate the retrieval performance.

```bash
python scripts/test_boolean_model.py
python scripts/test_vector_space_model.py
```

## License
This project is for educational purposes.

## Contributing
Contributions are welcome! Please feel free to open an issue or submit a pull request.
