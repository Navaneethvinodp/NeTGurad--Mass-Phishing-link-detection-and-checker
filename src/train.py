import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
import joblib
import os

def train_models():
    data_path = r"D:\Aaa Class\sem_7\final_project\colab project\final_project\data\dataset_phishing.csv"
    data = pd.read_csv(data_path)

    # Use HashingVectorizer instead of CountVectorizer
    n_features = 10000  # Choose an appropriate number of features
    vectorizer = HashingVectorizer(n_features=n_features)
    X_urls = vectorizer.fit_transform(data['url'])
    X_urls = X_urls.toarray()

    # Combine the URL features with the other features
    X_other_features = data.iloc[:, :-1].drop(columns=['url']).values
    X = np.hstack((X_other_features, X_urls))

    y = data.iloc[:, -1]

    # Encode categorical features
    for col in data.select_dtypes(include=['object']).columns:
        if col != 'url':
            le = LabelEncoder()
            X[:, data.columns.get_loc(col)] = le.fit_transform(X[:, data.columns.get_loc(col)])

    # If y is categorical, encode it too
    if y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)

    # Save the label encoder for y
    model_save_path = r"D:\Aaa Class\sem_7\final_project\colab project\final_project\models"
    os.makedirs(model_save_path, exist_ok=True)
    joblib.dump(le_y, os.path.join(model_save_path, 'label_encoder.pkl'))

    # Save the HashingVectorizer parameters
    joblib.dump(n_features, os.path.join(model_save_path, 'n_features.pkl'))

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display the results
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Save the trained model
    joblib.dump(model, os.path.join(model_save_path, 'model.pkl'))
    print("Model trained and saved successfully.")

# Call the function to train models
train_models()
