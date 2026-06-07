# Spam Detection Web App

A lightweight Flask web application that classifies messages (email/SMS) as **Spam** or **Ham** using a scikit-learn pipeline. The app includes a reusable serialized pipeline (`spam_classifier_pipeline.joblib`) and a simple web UI at `templates/index.html`.

This README focuses on features, architecture, setup, retraining, evaluation, deployment guidance, and troubleshooting.

Checklist
- Quick start (local development)
- Features and architecture overview
- How to retrain and evaluate the model
- Deployment recommendations and security notes

Key features
- Accurate, explainable pipeline:
  - Custom `TextPreprocessor` for cleaning, tokenization, stemming and stopword removal.
  - TF–IDF vectorization with uni- and bi-grams for richer feature representation.
  - Logistic Regression classifier with `class_weight='balanced'` to mitigate class imbalance.
- Interactive web UI:
  - Lightweight Flask form at `/` for pasting text and receiving instant predictions and confidence scores.
  - UI shows predicted label (Spam/Ham) and model confidence (probability).
- Robust data handling:
  - Flexible CSV loading with multiple fallback paths.
  - Validation and cleaning of missing or malformed rows in `prepare_data()`.
- Reproducible model storage:
  - Model serialized to `spam_classifier_pipeline.joblib` for fast startup and easy deployment.
- Easy to retrain and extend:
  - `app.py` exposes helper functions for data loading, preparation, pipeline building, training and persistence.

Repository structure
- `app.py` — Flask app and ML utilities (preprocessor, pipeline builder, training helpers).
- `spam_classifier_pipeline.joblib` — serialized trained pipeline (used for predictions).
- `mail_data.csv` — sample dataset (if present) used for training.
- `templates/index.html` — web UI template.

Quick start (development)
1. Create & activate a virtual environment (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies (create or update `requirements.txt` as shown below):
```powershell
python -m pip install -r requirements.txt
```
3. Download required NLTK data (if not already present):
```powershell
python - <<'PY'
import nltk
nltk.download('punkt')
nltk.download('stopwords')
PY
```
4. Run the app:
```powershell
python app.py
```
Open http://127.0.0.1:5000/ and paste text to see predictions.

Minimal recommended dependencies
Add these to `requirements.txt` (replace with exact versions from `pip freeze` for deployment):

Flask>=2.0
scikit-learn>=1.0
pandas>=1.3
nltk>=3.6
joblib>=1.0
numpy>=1.19

To generate an exact `requirements.txt` from your environment:
```powershell
python -m pip freeze > requirements.txt
```
Or generate a minimal list inferred from your imports (may miss some transitive runtime packages):
```powershell
python -m pip install pipreqs
pipreqs . --force
```

How the model works (high level)
1. Incoming text -> `TextPreprocessor`:
   - Removes URLs and email addresses, strips non-letters, lowercases text.
   - Tokenizes; optionally stems tokens using `PorterStemmer`, removes stopwords.
2. TF–IDF vectorizer converts preprocessed documents into numeric features (1–2 grams).
3. Logistic Regression predicts the class; `predict_proba()` is used by the UI to display confidence.

Files & functions to know in `app.py`
- `load_data(file_name='mail_data.csv')` — loads CSV, tries multiple paths.
- `prepare_data(data)` — validates columns, handles missing data, maps categories to labels.
- `build_pipeline()` — constructs the sklearn pipeline (preprocessor → TfidfVectorizer → LogisticRegression).
- `train_and_save_model(x_train, y_train, model_path='spam_classifier_pipeline.joblib')` — trains and saves a model.
- App route `/` expects form field `email_text` (POST) and returns the rendered HTML with prediction.

Retrain the model (example)
Run this in a Python REPL or a separate script — do not run inside the Flask process that is already serving the serialized model.
```python
from app import load_data, prepare_data, train_and_save_model
from sklearn.model_selection import train_test_split

data = load_data('mail_data.csv')
X, y = prepare_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = train_and_save_model(X_train, y_train, model_path='spam_classifier_pipeline.joblib')
```

Evaluate model (example)
```python
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, classification_report
pipeline = load('spam_classifier_pipeline.joblib')
y_pred = pipeline.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1:', f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

API / automated requests
- The web UI expects form field `email_text` at `/` (POST).
- Example using Python `requests`:
```python
import requests
resp = requests.post('http://127.0.0.1:5000/', data={'email_text': 'Win a free iPhone today!'})
print(resp.text)
```

Common issues & troubleshooting
- Missing or invalid `spam_classifier_pipeline.joblib`:
  - Ensure a valid model file exists in the project root. Retrain if needed (see Retrain section).
- NLTK LookupError:
  - Run the NLTK downloads shown earlier.
  - Note: `app.py` contains a non-standard token resource name `punkt_tab` in one download attempt. If you encounter tokenization errors, change that download call to use `'punkt'` instead of `'punkt_tab'` in your local copy of `app.py`.
- If Flask fails to start on port 5000 because of collisions, change port in `app.run()` or run:
```powershell
python -m flask run --port 5001
```

Production deployment suggestions
- Do NOT run Flask with `debug=True` in production.
- Use a production WSGI server:
  - Windows: use Waitress
  ```powershell
  python -m pip install waitress
  waitress-serve --listen=0.0.0.0:8000 app:app
  ```
  - Linux: use Gunicorn
- Containerization (Docker) is recommended for consistent environments.
- Pin package versions in `requirements.txt` for reproducible builds.
- Add logging, monitoring, and health checks.

Security & data privacy
- Avoid storing sensitive user data in plain text.
- If you log input text for debugging, sanitize or anonymize it.
- Add rate limiting and input size limits when exposing the API.

Improvements & roadmap
- Replace the simplistic stemmer with lemmatization (spaCy) for better token normalization.
- Add explainability (feature importance or LIME/SHAP) to surface why a message was classified as spam.
- Add unit tests and CI pipeline (GitHub Actions) to ensure reproducibility.
- Add model versioning and evaluation tracking (e.g., MLflow, DVC).

Contact
- E-mail: syeadahalzakir@gmail.com


