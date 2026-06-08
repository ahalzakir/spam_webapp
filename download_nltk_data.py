import nltk
import os

# Download NLTK data to project directory
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)

print(f"Downloading NLTK data to {nltk_data_path}...")
nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=False)
nltk.download('stopwords', download_dir=nltk_data_path, quiet=False)
print("NLTK data download complete!")

