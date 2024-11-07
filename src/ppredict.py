import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from sklearn.feature_extraction.text import HashingVectorizer
import joblib
import os

def crawl_and_collect_urls(base_url, num_urls=100):
    """Crawl the given URL and collect websites (both internal and external links)."""
    collected_urls = set()  # Use a set to avoid duplicates
    to_crawl = [base_url]  # List of URLs to crawl

    while to_crawl and len(collected_urls) < num_urls:
        current_url = to_crawl.pop(0)  # Take one URL to crawl
        try:
            response = requests.get(current_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract all anchor tags
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(current_url, href)  # Build the full URL

                # Validate if it's a legitimate URL and not an anchor or empty link
                parsed_href = urlparse(full_url)
                if parsed_href.scheme and parsed_href.netloc:
                    if full_url not in collected_urls and len(collected_urls) < num_urls:
                        collected_urls.add(full_url)
                        to_crawl.append(full_url)  # Add for further crawling
                    
                if len(collected_urls) >= num_urls:
                    break

        except Exception as e:
            print(f"Error while crawling {current_url}: {e}")
    
    return list(collected_urls)[:num_urls]  # Return only the first num_urls collected

def find_similar_websites(base_url, num_similar=50):
    """Search for similar websites on the internet by manipulating the URL."""
    parsed_url = urlparse(base_url)
    base_domain = parsed_url.netloc.split('.')[-2]  # Get the base domain (without TLD)
    
    # Generate some similar URL patterns based on common variations
    similar_patterns = [
        f"http://{base_domain}.com",
        f"http://www.{base_domain}.com",
        f"https://{base_domain}.net",
        f"http://{base_domain}.org",
        f"http://{base_domain}-secure.com",
        f"http://{base_domain}-verify.com",
        f"https://www.{base_domain}.co",
        f"https://{base_domain}.info",
        f"http://{base_domain}.online",
        f"https://www.secure-{base_domain}.com"
    ]

    # Expand by appending variations
    similar_websites = set()
    for pattern in similar_patterns:
        if len(similar_websites) >= num_similar:
            break
        try:
            response = requests.get(pattern)
            if response.status_code == 200:
                similar_websites.add(pattern)
        except:
            pass

    return list(similar_websites)[:num_similar]

# Load the trained model and vectorizer parameters
model_save_path = r"D:\Aaa Class\sem_7\final_project\colab project\final_project\models"
model = joblib.load(os.path.join(model_save_path, 'model.pkl'))
n_features = joblib.load(os.path.join(model_save_path, 'n_features.pkl'))
label_encoder = joblib.load(os.path.join(model_save_path, 'label_encoder.pkl'))

# Create a new HashingVectorizer with the same parameters
vectorizer = HashingVectorizer(n_features=n_features)

def preprocess_data(urls):
    """Preprocess and vectorize URLs for model prediction."""
    X_urls = vectorizer.transform(urls).toarray()
    
    # Adjust for missing features, if necessary
    additional_features = model.n_features_in_ - X_urls.shape[1]
    
    if additional_features > 0:
        X_additional = np.zeros((X_urls.shape[0], additional_features))
        X = np.hstack((X_urls, X_additional))
    else:
        X = X_urls
    
    return X

def predict_url_legitimacy(url, check_similar=False):
    """Crawl URLs starting from the given base URL and predict their legitimacy."""
    # Collect URLs by crawling the base URL
    collected_urls = crawl_and_collect_urls(url, num_urls=100)
    all_urls = [url] + collected_urls  # Include the original URL

    # Check for similar websites if requested
    if check_similar:
        similar_websites = find_similar_websites(url, num_similar=50)
        all_urls.extend(similar_websites)

    # Display URLs being checked
    print("\nChecking the following URLs for legitimacy:")
    for collected_url in all_urls:
        print(collected_url)

    # Preprocess the data for model prediction
    X = preprocess_data(all_urls)

    # Check the input feature size
    print(f"\nInput feature size: {X.shape[1]}")
    print(f"Model expects: {model.n_features_in_} features")

    # Predict using the trained model
    predictions = model.predict(X)

    # Map predictions back to original labels
    results = {url: label_encoder.inverse_transform([pred])[0] for url, pred in zip(all_urls, predictions)}

    return results

if __name__ == "__main__":
    input_url = input("Enter the URL to start crawling: ")
    check_similar = input("Do you want to search for similar websites as well? (yes/no): ").strip().lower() == 'yes'
    results = predict_url_legitimacy(input_url, check_similar=check_similar)

    # Prepare DataFrame for saving to CSV
    df_results = pd.DataFrame(list(results.items()), columns=['URL', 'Classification'])

    # Save results to a CSV file
    output_file_path = os.path.join(model_save_path, 'crawledd_classification_results.csv')
    df_results.to_csv(output_file_path, index=False)

    # Display the predictions
    print(f"\nClassification results saved to {output_file_path}.\n")
    print(df_results)
