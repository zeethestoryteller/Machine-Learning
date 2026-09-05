## 7. Named Entity Recognition & Simple NLP with spaCy

This final section moves beyond classical vectorization (BoW/TF-IDF) into a different kind of NLP tool: **spaCy**, a library that comes with pre-built language understanding — it already "knows" grammar and common entities without you having to train anything yourself.

### Setup

```python
# pip install spacy && python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load('en_core_web_sm')
```

- `en_core_web_sm` is a small, pre-trained English language model. "Pre-trained" is the key word — unlike TF-IDF or BoW (which you build from your own corpus), spaCy ships with a model already trained on large amounts of English text, so it already understands things like grammar structure and common named entities out of the box.

### Named Entity Recognition (NER)

```python
doc = nlp("Apple is looking at buying UK startup for $1 billion")
for ent in doc.ents:
    print(ent.text, ent.label_)   # Apple:ORG, UK:GPE, $1 billion:MONEY
```

**What's happening:** spaCy scans the sentence and automatically identifies "named entities" — real-world objects/concepts like organizations, locations, money amounts, dates, people, etc. — and tags them with a category label.

In this example:
- **"Apple"** → `ORG` (organization)
- **"UK"** → `GPE` (Geopolitical Entity — countries, cities, states)
- **"$1 billion"** → `MONEY`

This is incredibly useful for tasks like: extracting company names from news articles, pulling monetary figures from financial reports, or identifying locations mentioned in text — all without you having to write a single regex rule or train a classifier.

### Dependency Parsing

```python
for token in doc:
    print(token.text, token.dep_, token.head.text)
```

**What's happening:** For every single word (token) in the sentence, spaCy tells you:
- `token.text` — the word itself
- `token.dep_` — its grammatical *dependency relation* (e.g., is it the subject, object, modifier?)
- `token.head.text` — the word it grammatically depends on (its "parent" in the sentence structure)

For example, in "Apple is looking at buying," the word "looking" would likely be the head (main verb), and "Apple" would depend on it as the subject (`nsubj`). This builds a tree-like structure showing how all the words in a sentence relate to each other grammatically — useful for tasks like extracting "who did what to whom" from a sentence.

---

## Key Takeaways (Summary Table)

This is the document's cheat-sheet distillation of everything covered:

| Concept | Rule |
|---|---|
| TF-IDF | Default vectoriser for text; always inside pipeline |
| N-grams (1,2) | Almost always better than unigrams alone |
| Cosine similarity | Gold standard for document/item similarity |
| Image preprocessing | Resize → flatten → normalise before any sklearn model |
| Baseline | TF-IDF + LogisticRegression first, then iterate |

The overarching advice: **don't overthink your first model.** Start simple (TF-IDF + Logistic Regression, or pixel-flatten + Random Forest), get a baseline score, and only add complexity (bigrams, HOG, tuning) once you know your baseline number.

### Pen & Paper Example: Bag of Words Matrix (worked example from the doc)

**Input:** Two documents: "cat sat" and "cat and dog".

**1. Vocabulary (alphabetical):** `["and", "cat", "dog", "sat"]` — size 4.

**2. Vector for "cat sat":**
- and: 0, cat: 1, dog: 0, sat: 1 → **[0, 1, 0, 1]**

**3. Vector for "cat and dog":**
- and: 1, cat: 1, dog: 1, sat: 0 → **[1, 1, 1, 0]**

**Result Matrix:**
```
[[0, 1, 0, 1],
 [1, 1, 1, 0]]
```

This is the simplest possible illustration of BoW: build an alphabetical vocabulary across all documents, then for each document, count how many times each vocabulary word appears, in that fixed order. Every row is a document, every column is a vocabulary word, and the value is the count.

---

That's the entire document, front to back! We covered:
1. Text Representations (BoW, N-grams, TF-IDF, char n-grams, hashing)
2. CountVectorizer vs TfidfVectorizer (sklearn implementation)
3. Text Preprocessing (cleaning + lemmatization)
4. Text Classification Pipeline (Pipeline, cross-validation, GridSearchCV)
5. Cosine Similarity (math + code + worked example)
6. Image Processing for Classical ML (flatten, HOG)
7. NER & spaCy

Want a quick quiz to test yourself on any of these sections, or should we dig deeper into a specific topic?
