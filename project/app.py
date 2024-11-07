from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from sklearn.feature_extraction.text import HashingVectorizer
import joblib
import os
import random
import string
from urllib.parse import urlparse
app = Flask(__name__)

# Load the trained model and vectorizer parameters
model_path = r"D:\Aaa Class\sem_7\final_project\colab project\final_project\models"
model = joblib.load(os.path.join(model_path, 'model.pkl'))
n_features = joblib.load(os.path.join(model_path, 'n_features.pkl'))
label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))

vectorizer = HashingVectorizer(n_features=n_features)

# Function to validate and clean URLs
def validate_and_collect_urls(urls, max_urls):
    valid_urls = set()
    for url in urls:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                valid_urls.add(url)
            if len(valid_urls) >= max_urls:
                break
        except requests.RequestException:
            pass
    return list(valid_urls)

# Function to crawl and extract URLs from a webpage
def crawl_page_urls(base_url, max_urls=100):
    try:
        response = requests.get(base_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        page_urls = {urljoin(base_url, link['href']) for link in soup.find_all('a', href=True)}
        return validate_and_collect_urls(page_urls, max_urls)
    except requests.RequestException:
        return []

# Generate random similar website patterns based on base domain
def generate_random_websites(base_domain, num_similar=50):
    tlds = ['.com', '.net', '.org', '.info', '.co', '.online']
    random_websites = {f"http://{base_domain}{''.join(random.choices(string.ascii_lowercase + string.digits, k=5))}{random.choice(tlds)}"
                       for _ in range(num_similar * 2)}
    return validate_and_collect_urls(random_websites, num_similar)
common_suffixes = [
    'admin', 'login', 'logout', 'user', 'home', 'dashboard', 'profile',
    'settings', 'about', 'contact', 'help', 'terms', 'privacy',
    'products', 'services', 'cart', 'checkout', 'news', 'blog',
    'forum', 'search', 'register', 'reset', 'verify', 'subscribe',
    'support', 'feedback', 'downloads', 'notifications', 'gallery',
    'community', 'resources', 'events', 'testimonials', 'careers',
    'jobs', 'newsletters', 'articles', 'press', 'partners', 'faq',
    'legal', 'site-map', 'api', 'policy', 'cookies', 
    'login-failed', 'activation', 'confirmation', 'profile-edit', 
    'password-reset', 'track-order', 'wishlist', 'offers', 
    'promotions', 'settings-update', 'account', 'billing', 
    'subscriptions', 'history', 'analytics', 'reports', 
    'landing', 'newsletter-signup', 'home-delivery', 'purchases', 
    'preferences', 'community-guidelines', 'contact-us', 
    'media', 'portfolio', 'case-studies', 'tutorials', 
    'user-guide', 'maintenance', 'cookies-policy', 'terms-of-service', 
    'verify-email', 'invite', 'event-registration', 'member-area', 
    'discount', 'campaigns', 'success', 'error', 'archive', 
    'notifications', 'team', 'about-us', 'our-story', 
    'latest-updates', 'site-news', 'product-reviews', 'video', 
    'join', 'logout-success', 'login-help', 'secure-area', 
    'premium', 'user-settings', 'upload', 'resources-center',
    'referral', 'offers-and-promotions', 'service-status', 'payment',
    'billing-history', 'wishlist', 'gift-cards', 'rewards',
    'loyalty-program', 'help-center', 'documentation', 'guide',
    'video-tutorials', 'product-specifications', 'service-updates',
    'community-forum', 'events-calendar', 'user-analytics',
    'client-portal', 'downloads-center', 'trial', 'demo',
    'feature-request', 'terms-and-conditions', 'cookies-settings',
    'data-export', 'api-documentation', 'release-notes',
    'beta-testing', 'feedback-survey', 'case-studies', 'client-reviews',
    'error-log', 'scheduled-maintenance', 'contact-support', 
    'live-chat', 'tutorials-and-guides', 'setup', 'initial-setup',
    'system-requirements', 'technical-support', 'webinars',
    'online-training', 'invoices', 'previous-orders', 'subscription-plans',
    'user-profile', 'activity-log', 'search-results', 'advanced-search',
    'settings-general', 'settings-advanced', 'team-management',
    'project-management', 'bug-reports', 'status-updates',
    'account-settings', 'session-expired', 'verification', 'privacy-settings'
]
# New function to generate additional similar websites
def generate_similar_websites(original_url):
        parsed_url = urlparse(original_url)  
        base_url = parsed_url.netloc.split('.')[1]
        tlds = ['.com', '.net', '.org', '.info']
        
        similar_websites = []
        sim_web=[]
        for suffix in common_suffixes:
            tld = random.choice(tlds)
            generated_url = f"https://{base_url}{suffix}{tld} - not present"
            similar_websites.append(generated_url)
            
        
            #similar_websites.append(f"https://{base_url}{suffix}{tld}")
    
        
        return similar_websites




    #tlds = ['.com', '.net', '.org', '.info']  # List of top-level domains to append
    #similar_websites = []

    #for suffix in common_suffixes:
    #    tld = random.choice(tlds)
    #    similar_websites.append(f"{base_url}{suffix}{tld}")

    # Generate some similar URLs
    #for _ in range(50):  # Generate 5 similar URLs
    #    random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=7))
    #    tld = random.choice(tlds)
    #    s= {base_url}
    #   similar_websites.append(f"s.{random_suffix}{tld}")

    #return similar_websites

# Find similar websites based on the base domain
def find_similar_websites(base_url, num_similar=50):
    parsed_url = urlparse(base_url)
    base_domain = parsed_url.netloc.split('.')[-2]
    predefined_patterns = [
        f"http://{base_domain}.com",
        f"http://www.{base_domain}.com",
        f"https://{base_domain}.net",
        f"http://{base_domain}.org",
        f"http://{base_domain}-secure.com",
        f"http://{base_domain}-verify.com"
        f"http://{base_domain}.com",
        f"http://www.{base_domain}.com",
        f"https://{base_domain}.net",
        f"http://{base_domain}.org",
        f"http://{base_domain}-secure.com",
        f"http://{base_domain}-verify.com"
        f"http://{base_domain}.com",
        f"http://www.{base_domain}.com",
        f"https://{base_domain}.net",
        f"http://{base_domain}.org",
        f"http://{base_domain}-secure.com",
        f"http://{base_domain}-verify.com"
        f"http://{base_domain}.com",
        f"http://www.{base_domain}.com",
        f"https://{base_domain}.net",
        f"http://{base_domain}.org",
        f"http://{base_domain}-secure.com",
        f"http://{base_domain}-verify.com"
        f"http://{base_domain}.com",
        f"http://www.{base_domain}.com",
        f"https://{base_domain}.net",
        f"http://{base_domain}.org",
        f"http://{base_domain}-secure.com",
        f"http://{base_domain}-verify.com"
    ]
    return predefined_patterns + generate_random_websites(base_domain, num_similar)

# Preprocess URLs using the vectorizer
def preprocess_urls(urls):
    X_urls = vectorizer.transform(urls).toarray()
    additional_features = model.n_features_in_ - X_urls.shape[1]
    
    if additional_features > 0:
        X_additional = np.zeros((X_urls.shape[0], additional_features))
        X = np.hstack((X_urls, X_additional))
    else:
        X = X_urls
    return X

# Predict the legitimacy of a URL and its similar sites
def predict_url_legitimacy(url, check_similar=False):
    collected_urls = crawl_page_urls(url, max_urls=100)
    all_urls = [url] + collected_urls

    # Collect similar websites if requested
    if check_similar:
        similar_websites = find_similar_websites(url, num_similar=50)
        all_urls.extend(similar_websites)
    else:
        similar_websites = []

    # Generate random similar websites
    random_similar_websites = generate_similar_websites(url)

    # Preprocess URLs and make predictions
    X = preprocess_urls(all_urls)
    predictions = model.predict(X)
    results = {url: label_encoder.inverse_transform([pred])[0] for url, pred in zip(all_urls, predictions)}

    # Filter similar websites results
    similar_results = {sim_url: results.get(sim_url, "Not checked") for sim_url in similar_websites}

    return results, similar_results, random_similar_websites

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    input_url = request.form['url']
    check_similar = 'check_similar' in request.form

    results, similar_results, random_similar_websites = predict_url_legitimacy(input_url, check_similar=check_similar)
    
    return jsonify({
        'results': results,
        'similar_websites': similar_results,
        'random_similar_websites': random_similar_websites  # Added random websites to response
    })

if __name__ == "__main__":
    app.run(debug=True)
