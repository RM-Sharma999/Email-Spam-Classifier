def to_dense(X):
  return X.toarray()

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lt = WordNetLemmatizer()

custom_stopwords = set([
    'fw', 'fwd', 'subject', 'cc', 'bcc',
    'message', 'original', 'reply', 'sent', 'mail', 'email', 'forwarded',
    'content', 'type', 'text', 'plain', 'html', 'charset', 'encoding',
    'format', 'quoted', 'printable', '7bit', 'part', 'boundary',
    'n', 'rn', 'nn', 'nbsp', 'br'
])

stop_words = set(stopwords.words('english')).union(custom_stopwords)  # Load all stopwords once

# Custom Function to apply Text Preprocessing
def transform_text(text):
  # Convert text to lowercase
  text = text.lower()

  # Remove email addresses and URLs
  text = re.sub(r'\S+@\S+', '', text)
  text = re.sub(r'https?://\S+|www\.\S+', '', text)

  # Remove leftover 'http', 'https', 'www' tokens
  text = re.sub(r'\bhttp\b|\bhttps\b|\bwww\b', '', text)

  # Remove HTML tags
  text = re.sub(r'<.*?>', '', text)

  # Remove newline/tab characters
  text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

  # Split text into individual words
  tokens = nltk.word_tokenize(text)

  # Keep only alphanumeric words (no punctuation or special characters)
  tokens = [token for token in tokens if token.isalnum()]

  # Remove stopwords
  filtered = [token for token in tokens if token not in stop_words]

  # Apply lemmatization to reduce words to their base form
  lemmatized = [lt.lemmatize(token) for token in filtered]

  return " ".join(lemmatized).strip()  # Return cleaned text as a single string

# Custom transformer to apply transform_text to text data
class TextCleaner(BaseEstimator, TransformerMixin):
  def fit(self, X, y = None):
    return self

  def transform(self, X):
    X = X.copy()
    if isinstance(X, pd.DataFrame):
      # Join all columns if more than one
      if X.shape[1] > 1:
        X = X.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
      else:
        X = X.iloc[:, 0]  # Extract the single column as Series
    elif isinstance(X, np.ndarray):
      X = pd.Series(X.ravel())
    return X.apply(transform_text)