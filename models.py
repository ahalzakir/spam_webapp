# models.py
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.base import BaseEstimator, TransformerMixin


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stem=True, remove_stopwords=True):
        self.stemmer = PorterStemmer() if stem else None
        self.stopwords = set(stopwords.words('english')) if remove_stopwords else None

    def clean_text(self, text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.lower()

    def tokenize_and_stem(self, text):
        try:
            tokens = word_tokenize(text)
        except LookupError:
            print("Downloading punkt resource...")
            nltk.download('punkt', quiet=True)
            tokens = word_tokenize(text)

        if self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        return ' '.join(tokens)

    def transform(self, x, y=None):
        return [self.tokenize_and_stem(self.clean_text(str(text))) for text in x]

    def fit(self, x, y=None):
        return self