# built-in Python modules used for handling and manipulating text.

### `re` (Regular Expressions): 
Provides regular expression matching operations. It allows you to search, match, split, or replace text patterns (such as finding URLs, removing special characters, or validating email formats) using specific search patterns.

**Common operations**

* **re.search():** Scans through a string looking for the first location where the regular expression pattern produces a match, returning a match object or `None`.
* **re.match():** Checks for a match only at the beginning of the string.
* **re.findall():** Finds all non-overlapping matches of a pattern in a string and returns them as a list of strings.
* **re.finditer():** Returns an iterator yielding match objects over all non-overlapping matches for a pattern in a string.
* **re.sub():** Replaces occurrences of the pattern in a string with a specified replacement string or function (`re.sub(pattern, repl, string)`).
* **re.split():** Splits a string by the occurrences of a pattern, returning a list containing the resulting substrings.
* **re.compile():** Compiles a regular expression pattern into a regex object, which can be reused efficiently for multiple operations using methods like `.search()` or `.findall()`.

### string: 
Contains a collection of useful string constants and classes, such as ascii letters, digits, and punctuation marks (e.g., string.punctuation gives you !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~), which are commonly used for text cleaning and validation.

* **`string.punctuation`**: A string constant containing all standard punctuation characters: `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`.
* **`str.maketrans(x, y, z)`**: Creates a translation mapping table.
  * The first two arguments (`''`, `''`) are used for mapping character-to-character replacements (which are empty here since we aren't replacing characters).
  * The third argument (`string.punctuation`) specifies a string of characters that should be **deleted** entirely.

* **`text.translate(...)`**: Applies the translation table to the string, stripping out every character present in `string.punctuation` in one fast C-level pass without needing slow regular expression loops.




```python
import re
import string


raw_texts = [
    'the cat sat on the mat',
    'the dog sat on the log',
    'the cat and the dog play'
]

def clean_text(text):
    text = text.lower()                               # lowercase
    text = re.sub(r'http\S+', '', text)              # remove URLs
    text = re.sub(r'@\w+', '', text)                 # remove mentions
    text = re.sub(r'#', '', text)                     # remove hashtag symbol
    text = text.translate(str.maketrans('','',string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()         # normalise whitespace
    return text

texts = [clean_text(t) for t in raw_texts]

```

# Optional: NLTK lemmatisation

This code performs **lemmatization** on text, which reduces words down to their root or dictionary form (e.g., changing "running" or "runs" to "run").

* **`import nltk`**: Imports the Natural Language Toolkit library, a popular platform for building Python programs to work with human language data.
* **`from nltk.stem import WordNetLemmatizer`**: Imports the `WordNetLemmatizer` class, which uses the WordNet lexical database to find the base forms of words based on their intended meanings.
* **`from nltk.tokenize import word_tokenize`**: Imports the tokenizer function that splits a sentence or document string into individual word and punctuation tokens.
* **`nltk.download(...)`**: Downloads essential NLTK datasets (`punkt` for tokenization, `wordnet` for the lemmatization dictionary, and `stopwords` for common words) required for the text processing tasks to run locally.
* **`lem = WordNetLemmatizer()`**: Instantiates the lemmatizer object so its `.lemmatize()` method can be called.
* **`def lemmatise(text):`**: Defines a function that takes a text string as input.
* **`word_tokenize(text)`**: Splits the input string into a list of individual words/tokens.
* **`lem.lemmatize(w)`**: Converts each individual token `w` into its base lemma form.
* **`' '.join(...)`**: Combines the list of lemmatized words back together into a single continuous space-separated string and returns the result.

```python
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Sample text with different word forms and tenses
text = "The cats were running faster than the mice, and they played joyfully."

# Tokenize the text into individual words
words = word_tokenize(text)

# Apply lemmatization to each word
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

# Rejoin into a single sentence
lemmatized_text = 'ژگی'.join(lemmatized_words) # standard space join
lemmatized_text = ' '.join(lemmatized_words)

print("Original Text:", text)
print("Lemmatized:", lemmatized_text)

```

**Output:**

```text
Original Text: The cats were running faster than the mice, and they played joyfully.
Lemmatized: The cat were running faster than the mouse , and they played joyfully.

```

*(Note: To convert verbs like "running" or "played" into their base infinitive verbs like "run" or "play", you can pass a Part-of-Speech tag like `pos='v'` to `lemmatizer.lemmatize(word, pos='v')`.)*
