import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Flatten
from sklearn.preprocessing import StandardScaler, LabelEncoder

file_path = r"D:\Aaa Class\sem_7\final_project\colab project\final_project\data\dataset_phishing.csv"
# Load the dataset

# Load your dataset (ensure it's cleaned and well-structured)
data = pd.read_csv(file_path)  # Replace with your actual dataset



# Inspect for any missing values and drop rows/columns with too many NaNs
data = data.dropna()

# Handle categorical values (like `status` which may be 'legitimate' or 'phishing')
label_encoder = LabelEncoder()
data['status'] = label_encoder.fit_transform(data['status'])  # 0 for legitimate, 1 for phishing

# Preprocess structured features for Random Forest
structured_features = ['length_url', 'length_hostname', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_qm', 'nb_and', 'nb_or', 
                       'nb_eq', 'nb_underscore', 'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon', 
                       'nb_comma', 'nb_semicolumn', 'nb_dollar', 'nb_space', 'nb_www', 'nb_com', 'nb_dslash', 
                       'http_in_path', 'https_token', 'ratio_digits_url', 'ratio_digits_host', 'punycode', 'port',
                       'tld_in_path', 'tld_in_subdomain', 'abnormal_subdomain', 'nb_subdomains', 'prefix_suffix',
                       'random_domain', 'shortening_service', 'path_extension', 'nb_redirection', 'nb_external_redirection']

# Ensure that all features are numerical (clean up categorical if necessary)
X_structured = data[structured_features]
y = data['status']

# Scale the structured features
scaler = StandardScaler()
X_structured_scaled = scaler.fit_transform(X_structured)

# Split data for structured features
X_train_structured, X_test_structured, y_train, y_test = train_test_split(X_structured_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_structured, y_train)

# Predict and evaluate Random Forest
rf_predictions = rf_model.predict(X_test_structured)
print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))

# Preprocessing URLs for CNN
def preprocess_urls(urls):
    max_len = 100  # Set max length for URLs to ensure uniform size
    char_to_int = {chr(i): i for i in range(32, 127)}  # ASCII mapping for characters
    int_urls = [[char_to_int.get(char, 0) for char in url] for url in urls]  # Convert chars to ints
    int_urls = tf.keras.preprocessing.sequence.pad_sequences(int_urls, maxlen=max_len, padding='post', truncating='post')
    return np.array(int_urls)

X_urls = preprocess_urls(data['url'])
X_train_urls, X_test_urls, _, _ = train_test_split(X_urls, y, test_size=0.2, random_state=42)

# Build CNN model for URL text analysis
cnn_model = Sequential([
    Embedding(input_dim=128, output_dim=64, input_length=100),  # 128 ASCII characters, embedding dimension 64
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (1 for phishing, 0 for legitimate)
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train CNN model
cnn_model.fit(X_train_urls, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Predict and evaluate CNN
cnn_predictions = (cnn_model.predict(X_test_urls) > 0.5).astype(int)
print("CNN Classification Report:\n", classification_report(y_test, cnn_predictions))

# Ensemble: Combine the predictions from Random Forest and CNN
rf_probabilities = rf_model.predict_proba(X_test_structured)[:, 1]  # Get probabilities from RF
cnn_probabilities = cnn_model.predict(X_test_urls).flatten()  # Get probabilities from CNN

# Average the probabilities for an ensemble prediction
ensemble_probabilities = (rf_probabilities + cnn_probabilities) / 2
ensemble_predictions = (ensemble_probabilities > 0.5).astype(int)

# Final evaluation of the ensemble
print("Ensemble Model Classification Report:\n", classification_report(y_test, ensemble_predictions))

