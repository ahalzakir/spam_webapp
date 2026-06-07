import os
import pandas as pd
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import joblib

# text preprocessor
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


             print("Downloading punkt_tab resource...")
             nltk.download('punkt_tab', quiet=True)
             tokens = word_tokenize(text)

        if self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        return ' '.join(tokens)

    def transform(self, x, y=None):
        # Handle potential non-string inputs

        #apply the cleaning and tokenization to each element in X
        return [self.tokenize_and_stem(self.clean_text(str(text))) for text in x]

    def fit(self, x, y=None):
        return self

def load_data(file_name='mail_data.csv'):

    possible_paths = [
        file_name,
        os.path.join('data', file_name),
        os.path.join('dataset', file_name),
        os.path.join('content', file_name)
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                data = pd.read_csv(path)
                print(f"Dataset loaded successfully from: {path}")
                return data
            except Exception as e:
                print(f"Error loading {path}: {str(e)}")
                continue

    raise FileNotFoundError(f"Could not find {file_name} in any of these locations: {possible_paths}")

def prepare_data(data):
    """Data preparation with validation"""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    required_columns = {'Category', 'Message'}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # For handling missing values - replace None/NaN with empty string
    data = data.where((pd.notnull(data)), '')

    # Validate categories
    valid_categories = {'spam', 'ham'}
    unique_cats = set(data['Category'].unique())
    # Ensure all unique categories are either 'spam', 'ham', or an empty string
    if not (unique_cats - {'', *valid_categories}) == set():
         invalid = unique_cats - {'', *valid_categories}
         raise ValueError(f"Invalid categories found: {invalid}. Expected: {valid_categories} (and optionally empty string)")

    # Handle potential empty strings in 'Category' after .where
    data['Label'] = data['Category'].apply(lambda x: 0 if x == 'spam' else (1 if x == 'ham' else -1)) # Use -1 or similar for invalid/missing
    # Filter out rows where Category was missing/invalid if needed, or handle downstream
    data = data[data['Label'] != -1] # Remove rows with invalid/missing categories
    return data['Message'], data['Label']

def build_pipeline():

    pipeline = Pipeline([
        ('preprocessor', TextPreprocessor()),
        ('vectorizer', TfidfVectorizer(
            min_df=5,
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english')),
        ('classifier', LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            max_iter=1000))
    ])
    return pipeline

def train_and_save_model(x_train, y_train, model_path='spam_classifier_pipeline.joblib'):

    print("\nTraining model...")
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    joblib.dump(pipeline, model_path)
    print(f"Model trained and saved to {model_path}")
    return pipeline

def load_model(model_path='spam_classifier_pipeline.joblib'):

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            return None
    return None # Return None if model file doesn't exist

from flask import Flask, render_template, request
app = Flask(__name__)

# Load model
model = joblib.load('spam_classifier_pipeline.joblib')


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = 0
    probability = 0

    if request.method == 'POST':
        email_text = request.form['email_text']

        # Get prediction and probabilities
        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]

        result = "Spam" if prediction == 0 else "Ham"
        probability = probabilities[0] if prediction == 0 else probabilities[1]
        confidence = round(max(probabilities) * 100, 2)

    return render_template(
        'index.html',
        result=result,
        confidence=confidence,
        probability=max(probability, 1 - probability)  # Show highest class probability
    )


if __name__ == '__main__':
    app.run(debug=True)
