import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
import joblib
import os
from urllib.parse import urlparse

def crawl_and_find_related_urls(base_url, max_urls=100):
    visited_urls = set()
    to_visit = [base_url]
    related_urls = []

    while to_visit and len(related_urls) < max_urls:
        current_url = to_visit.pop(0)
        if current_url in visited_urls:
            continue

        visited_urls.add(current_url)
        try:
            response = requests.get(current_url, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = requests.compat.urljoin(current_url, href)
                if full_url.startswith('http') and full_url not in visited_urls:
                    related_urls.append(full_url)
                    to_visit.append(full_url)
                    if len(related_urls) >= max_urls:
                        break

        except Exception as e:
            print(f"Error while crawling {current_url}: {e}")

    return related_urls[:max_urls]

# Load the trained model and vectorizer parameters
model_save_path = r"D:\Aaa Class\sem_7\final_project\colab project\final_project\models"
model = joblib.load(os.path.join(model_save_path, 'model.pkl'))
n_features = joblib.load(os.path.join(model_save_path, 'n_features.pkl'))
label_encoder = joblib.load(os.path.join(model_save_path, 'label_encoder.pkl'))

# Create a new HashingVectorizer with the same parameters
vectorizer = HashingVectorizer(n_features=n_features)

def preprocess_data(urls):
    X_urls = vectorizer.transform(urls).toarray()
    additional_features = model.n_features_in_ - X_urls.shape[1]
    
    if additional_features > 0:
        X_additional = np.zeros((X_urls.shape[0], additional_features))
        X = np.hstack((X_urls, X_additional))
    else:
        X = X_urls
    
    return X

def predict_urls(urls):
    X = preprocess_data(urls)
    print(f"Input feature size: {X.shape[1]}")
    print(f"Model expects: {model.n_features_in_} features")
    
    predictions = model.predict(X)
    results = {url: label_encoder.inverse_transform([pred])[0] for url, pred in zip(urls, predictions)}
    return results

def get_domain(url):
    return urlparse(url).netloc

if __name__ == "__main__":
    input_url = input("Enter the URL to check: ")
    related_urls = crawl_and_find_related_urls(input_url, max_urls=100)
    all_urls = [input_url] + related_urls
    
    results = predict_urls(all_urls)
    
    # Prepare DataFrame for saving to CSV
    df_results = pd.DataFrame(list(results.items()), columns=['URL', 'Classification'])
    df_results['Domain'] = df_results['URL'].apply(get_domain)
    
    # Group by domain and get the most common classification
    domain_results = df_results.groupby('Domain').agg({
        'Classification': lambda x: x.value_counts().index[0],
        'URL': 'count'
    }).rename(columns={'URL': 'Count'}).reset_index()
    
    # Save results to CSV files
    output_file_path = os.path.join(model_save_path, 'url_classification_results.csv')
    df_results.to_csv(output_file_path, index=False)
    print(f"\nURL classification results saved to {output_file_path}")
    
    domain_output_file_path = os.path.join(model_save_path, 'domain_classification_results.csv')
    domain_results.to_csv(domain_output_file_path, index=False)
    print(f"Domain classification results saved to {domain_output_file_path}\n")
    
    # Display the domain classification results
    print("Domain Classification Results:")
    print(domain_results)