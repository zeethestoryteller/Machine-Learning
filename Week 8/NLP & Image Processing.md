# Learning Objectives
>Understand Bag-of-Words, N-grams, and TF-IDF
>
>Build text classification pipelines
>
>Use cosine similarity for search and recommendations
>
>Convert images to feature vectors for classical ML
---
# Text Representations
Text representation refers to the process of converting raw text into numerical feature vectors that machine learning algorithms can process. Because models require numbers rather than strings, various text representation techniques capture word frequencies, rare informative terms, or sequential context.

* **Bag of Words (BoW):** Counts the raw occurrences of each word in a document to form a vocabulary-sized vector, ignoring word order entirely.
  >Example: For the corpus ["the cat sat", "the dog sat"], the vocabulary is ["cat", "dog", "sat", "the"]. The vector for "the cat sat" is [1, 0, 1, 1] (ignoring word order).
* **N-grams:** Extends BoW by counting sequences of adjacent words (e.g., bigrams or trigrams) to partially preserve local word order and capture phrases like "not good".
  >Example: Using bigrams (range=2) on "not good", the extracted tokens include the combined sequence "not good" rather than just individual words, helping capture sentiment or negation.
* **TF-IDF (Term Frequency-Inverse Document Frequency):** Weights term frequency by document rarity, dampening the impact of extremely common words while boosting rare, informative terms.
  >Example: In a technical corpus where the word "algorithm" appears frequently in a single document but rarely across others, its high TF-IDF score will highlight its importance for that specific document
* **Character N-grams:** Analyzes character-level sequences instead of full words, which helps handle typos and morphologically rich languages.
  >Example: The word "apple" broken into character bigrams generates ["ap", "pp", "pl", "le"], which helps models handle misspellings, suffixes, and morphologically rich terms.
* **Hashing Vectorizer:** Maps words directly to a fixed-size index using a hash function, trading off interpretability for memory efficiency on massive corpora.
  >Example: The word "matrix" is passed through a hash function (e.g., MurmurHash) to map directly to index 425 in a vector of fixed length $2^{16}$, saving memory on massive datasets.


**TF-IDF Formula**
```
TF(t,d)  = count(t in d) / total_words(d)       # term frequency
IDF(t)   = log(N / df(t)) + 1                   # inverse document frequency
                                              # N = corpus size or total no of documents, df = no of docs containing t
TF-IDF(t,d) = TF(t,d) × IDF(t)
```
>Common words (high df) get lower IDF → lower weight. Rare but informative words get higher IDF → higher weight.


| Method | Idea | Dimensionality | Preserves order |
| --- | --- | --- | --- |
| Bag of Words | Count word occurrences | V (vocab size) | No |
| N-grams | Count word sequences | $V^n$ (large) | Partially |
| TF-IDF | Weight by document rarity | V | No |
| Char n-grams | Character-level sequences | Large | Partially |
| Hashing | Hash word to index | Fixed | No |

---
# CountVectorizer 
converts a collection of text documents into a numerical matrix of token counts through a systematic text-processing pipeline:

* **Lowercasing:** Converts all characters to lowercase (if `lowercase=True`) so that words like "Cat" and "cat" are treated as the same token.
* **Tokenization:** Breaks the text down into individual words or tokens based on a specified regular expression pattern (e.g., `token_pattern=r'(?u)\b\w\w+\b'` to extract words with at least two characters).
* **Vocabulary Building:** Scans the entire corpus to learn a unique vocabulary dictionary mapping each distinct token to a specific column index (e.g., `{'cat': 0, 'dog': 1, ...}`). It also applies pruning rules like `min_df`, `max_df`, and `max_features` to filter out terms that are too rare, too common, or exceed the vocabulary size limit.
* **Stop Word Removal:** Excludes common words (like "and", "the", "is") if configured with `stop_words='english'`.
* **Matrix Encoding:** Counts the raw frequency of each vocabulary word within each individual document, constructing a sparse matrix of shape `(n_documents, n_vocabulary_size)`.

*Example:* For the corpus `['the cat sat on the mat', 'the dog sat on the log']`, a `CountVectorizer` with English stop-word removal builds a vocabulary of unique content words (`['cat', 'dog', 'log', 'mat', 'sat']`) and counts their exact occurrences per document to produce the final sparse feature vectors.

# CountVectorizer vs TfidfVectorizer



```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    'the cat sat on the mat',
    'the dog sat on the log',
    'the cat and the dog play'
]

# Bag of Words
bow = CountVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),      # unigrams + bigrams
    max_features=10000,
    min_df=2,                # ignore terms in < 2 documents
    max_df=0.95,             # ignore terms in > 95% of documents
    token_pattern=r'(?u)\b\w\w+\b'  # at least 2 chars
)
X_bow = bow.fit_transform(corpus)
print(bow.vocabulary_)       # {word: index}
print(bow.get_feature_names_out())

# TF-IDF
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True,       # use 1+log(tf) instead of tf (reduces extremes)
    norm='l2'                # normalise each document vector to unit length
)
X_tfidf = tfidf.fit_transform(corpus)

# Inspect top terms for a document
doc_idx = 0
feature_names = tfidf.get_feature_names_out()
doc_scores = X_tfidf[doc_idx].toarray().flatten()
top_idx = doc_scores.argsort()[-10:][::-1]
print([(feature_names[i], doc_scores[i]) for i in top_idx])
```

**CountVectorizer Parameters**

* **`lowercase=True`:** Converts all text characters to lowercase before tokenization, ensuring words with different cases (e.g., "Cat" and "cat") are mapped to the same vocabulary index.
* **`stop_words='english'`:** Automatically filters out common, low-information English words (like "and", "the", "is") using a built-in stop-word list.
* **`ngram_range=(1, 2)`:** Specifies the lower and upper boundaries of the n-gram range to extract, capturing both single words (unigrams) and adjacent two-word sequences (bigrams).
* **`max_features=10000`:** Restricts the vocabulary size to only the top $10,000$ most frequent terms, discarding less common words to manage memory and computational overhead.
* **`min_df=2`:** Ignores terms that appear in fewer than $2$ documents, filtering out rare typos or idiosyncratic tokens.
* **`max_df=0.95`:** Ignores terms that appear in more than $95\%$ of the documents, filtering out corpus-specific stop words that lack discriminative power.
* **`token_pattern=r'(?u)\b\w\w+\b'`:** Uses a regular expression to define what constitutes a valid token, requiring words to consist of alphanumeric characters and be at least 2 characters long.

---

**TfidfVectorizer Parameters**

* **`sublinear_tf=True`:** Replaces the raw term frequency ($tf$) with $1 + \log(tf)$ to dampen the disproportionate impact of words that appear excessively often within a single document.
* **`norm='l2'`:** Applies L2 normalization to each document vector, scaling its Euclidean norm to $1$ so that document length doesn't bias similarity metrics.
