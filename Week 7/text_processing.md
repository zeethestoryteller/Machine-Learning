
# **Text Preprocessing**

## `CountVectorizer` vs `HashingVectorizer`

Vectorizers are used to convert a collection of text documents to a vector representation, thus helping in preprocessing them before applying any model on these text documents.

`CountVectorizer` and `HashingVectorizer` both perform the task of vectorizing the text documents. However, there are some differences among them.

One difference is that `HashingVectorizer` does not store the resulting vocabulary (i.e. the unique tokens). Hence, it can be used to learn from data that does not fit into the computer’s main memory. Each mini-batch is vectorized using `HashingVectorizer` so as to guarantee that the input space of the estimator has always the same dimensionality.

With `HashingVectorizer`, each token directly maps to a pre-defined column position in a matrix. For example, if there are 100 columns in the resultant (vectorized) matrix, each token (word) maps to 1 of the 100 columns. The mapping between the word and the position in matrix is done using hashing.

In other words, in `HashingVectorizer`, each token transforms to a column position instead of adding to the vocabulary. Not storing the vocabulary is useful while handling large data sets. This is because holding a huge token vocabulary comprising of millions of words may be a challenge when the memory is limited.

Since `HashingVectorizer` does not store vocabulary, its object not only takes lesser space, it also alleviates any dependence with function calls performed on the previous chunk of data in case of incremental learning.

### Example

Let us take some sample text documents and vectorize them, first using CountVectorizer and then HashingVectorizer.

```python
text_documents = ['The well-known saying an apple a day keeps the doctor away has a very straightforward, literal meaning, that the eating of fruit maintains good health.',
                  'The proverb first appeared in print in 1866 and over 150 years later is advice that we still pass down through generations.',
                  'British apples are one of the nations best loved fruit and according to Great British Apples, we consume around 122,000 tonnes of them each year.',
                  'But what are the health benefits, and do they really keep the doctor away?']

```

### 1. CountVectorizer

We will first import the library and then create an object of CountVectorizer class.

```python
from sklearn.feature_extraction.text import CountVectorizer
c_vectorizer = CountVectorizer()

```

We will now use this object to vectorize the input text documents using the function `fit_transform()`.

```python
X_c = c_vectorizer.fit_transform(text_documents)

```

```python
X_c.shape

```

Here, 66 is the size of the vocabulary.

We can also see the vocabulary using `vocabulary_` attribute.

```python
c_vectorizer.vocabulary_

```

Following is the representation of four text documents.

```python
print(X_c)

```

---

### HashingVectorizer

Let us now see how `HashingVectorizer` is different from `CountVectorizer`.

We will create an object of HashingVectorizer. While creating the object, we need to specify the number of features we wish to have in the feature matrix.

```python
from sklearn.feature_extraction.text import HashingVectorizer

```

Let us create an object of `HashingVectorizer` class. An important parameter of this class is `n_features`. It declares the number of features (columns) in the output feature matrix.

Note: Small numbers of features are likely to cause hash collisions, but large numbers will cause larger coefficient dimensions in linear learners.

```python
h_vectorizer= HashingVectorizer(n_features=50)

```

Let's perform hashing vectorization with `fit_transform`.

```python
X_h = h_vectorizer.fit_transform(text_documents)

```

Let us examine the shape of the transformed feature matrix. The number of columns in this matrix is equal to the `n_features` attribute we specified.

```python
X_h.shape

```

Let's print the representation of the first example.

```python
print(X_h[0])

```

Overall, `HashingVectorizer` is a good choice if we are falling short of memory and resources, or we need to perform incremental learning. However, `CountVectorizer` is a good choice if we need to access the actual tokens.

---

# **Combining preprocessing and fitting in Incremental Learning**

### (`HashingVectorizer` along with `SGDClassifier`)

We will now use a dataset containing a textual feature that requires preprocessing using a vectorizer. Since we wish to perform incremental learning using `partial_fit()`, we will preprocess (i.e., vectorize) the dataset feature using `HashingVectorizer` and then we will incrementally fit it.

### 1. Downloading the dataset

Below, we download a dataset from UCI ML datasets' library. (Instead of downloading, unzipping and then reading, we are directly reading the zipped csv file. For that purpose, we are making use of `urllib.request`, `BytesIO` and `TextIOWrapper` classes.)

This is a sentiment analysis dataset. There are only two columns in the dataset. One for the textual review and the other for the sentiment.

```python
import pandas as pd
from io import StringIO, BytesIO, TextIOWrapper
from zipfile import ZipFile
import urllib.request

resp = urllib.request.urlopen('[https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip](https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip)')
zipfile = ZipFile(BytesIO(resp.read()))

data = TextIOWrapper(zipfile.open('sentiment labelled sentences/amazon_cells_labelled.txt'), encoding='utf-8')

df = pd.read_csv(data, sep = '\t')
df.columns = ['review', 'sentiment']

```

## 2. Exploring the dataset.

Let's explore the dataset a bit.

```python
df.head()

```

```python
df.tail()

```

```python
df.info()

```

```python
df.describe()

```

```python
df.loc[:, 'sentiment'].unique()

```

As we can see,

* There are 999 samples in the dataset.
* The possible classes for sentiment are 1 and 0.

## 4. Splitting data into train and test

```python
from sklearn.model_selection import train_test_split

```

```python
X = df.loc[:, 'review']

```

```python
y= df.loc[:, 'sentiment']

```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

```

```python
X_train.shape

```

```python
y_train.shape

```

## 5. Preprocessing

Since the data is textual, we need to vectorize it. In order to perform incremental learning, we will use HashingVectorizer.

```python
from sklearn.feature_extraction.text import HashingVectorizer
vectorizer = HashingVectorizer()

```

## 6. Creating an instance of the SGDClassifier

```python
from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(penalty='l2',loss='hinge')

```

## 7. Iteration 1 of partial_fit()

We will assume we do not have sufficient memory to handle all the 799 samples in one go for training purpose. So, we will take the first 400 samples from the training data and `partial_fit` our classifier.

Another use case of partial_fit here could also be a scenario where we only have 400 samples available at a time. So, we fit our classifier with them. However, we `partial_fit` it, to have the possibility of training it with more data later whenever that becomes available.

```python
X_train_part1_hashed = vectorizer.fit_transform(X_train[0:400])
y_train_part1 = y_train[0:400]

```

```python
import numpy as np
all_classes = np.unique(df.loc[:, 'sentiment']) #we need to mention all classes in the first iteration of partial_fit()

```

```python
classifier.partial_fit(X_train_part1_hashed, y_train_part1, classes=all_classes)

```

Let us now use this classifier on our test data that we had kept aside earlier.

```python
X_test_hashed = vectorizer.transform(X_test) #first we will have to preprocess the X_test with the same vectorizer that was fit on train data.

```

```python
test_score = classifier.score(X_test_hashed, y_test)
print("Test score: ", test_score)

```

Note: We can also store this classifier using pickle object and can access it later.

# 8. Iteration 2 of partial_fit()

We will now assume that more data became available. So, we will fit the same classifier with more data and observe if our test score improves.

```python
X_train_part2_hashed = vectorizer.transform(X_train[400:])
y_train_part2 = y_train[400:]

```

```python
classifier.partial_fit(X_train_part2_hashed, y_train_part2)

```

```python
test_score = classifier.score(X_test_hashed, y_test)
print("Test score: ", test_score)

```

We see that our test score has improved after we fed more data to the classifier in the second iteration of `partial_fit()`.

For a more elaborate example, refer: [Scikit-Learn Out-of-core Classification Example](https://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html#sphx-glr-auto-examples-applications-plot-out-of-core-classification-py)

---

# TF-IDF

Term Frequency: TF of a term or word is the number of times the term appears in a document compared to the total number of words in the document.

$$\text{TF} = \frac{\text{Number of times the term appears in the document}}{\text{Total number of terms in the document}}$$

Inverse Document Frequency: IDF of a term reflects the proportion of documents in the corpus that contain the term. Words unique to a small percentage of documents receive higher importance values than words common across all documents.

$$\text{IDF} = \log \left(\frac{\text{Numbers of documents in the corpus} + 1}{\text{Numbers of documents in the corpus containing the term} + 1}\right)$$

The TF-IDF of a term is calculated by multiplying TF and IDF scores:


$$\text{TF-IDF} = \text{TF} \times \text{IDF}$$

### Example

|  | Corpus |
| --- | --- |
| r1 | “this is a a sample” |
| r2 | “this is another another example example example” |

## Calculate TF-IDF for the word "this"

* $\text{TF}(\text{"this" in r1}) = \frac{1}{5} = 0.2$
* $\text{TF}(\text{"this" in r2}) = \frac{1}{7} \approx 0.14$
* $\text{IDF}(\text{this}) = \log\left(\frac{3}{3}\right) = 0$
* $\text{TF-IDF in r1} = 0.2 \times 0 = 0$
* $\text{TF-IDF in r2} = 0.14 \times 0 = 0$

## Calculate TF-IDF for the word "example"

* $\text{TF}(\text{"example" in r1}) = \frac{0}{5} = 0$
* $\text{TF}(\text{"example" in r2}) = \frac{3}{7} \approx 0.429$
* $\text{IDF}(\text{example}) = \log\left(\frac{3}{2}\right) \approx 0.176$
* $\text{TF-IDF in r1} = 0 \times 0.176 = 0$
* $\text{TF-IDF in r2} = 0.429 \times 0.176 \approx 0.0755$

## Calculate TF-IDF for "sample"

* $\text{TF-IDF}(\text{"sample" in r1}) = 0.0352$
* $\text{TF-IDF}(\text{"sample" in r2}) = 0$

```

```
