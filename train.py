# train.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

# Create directories if they don't exist
os.makedirs("app/model", exist_ok=True)
os.makedirs("app/data", exist_ok=True)

# Load dataset
df = pd.read_csv("app/data/data.csv")

# Print column details for debugging
print(f"Total Columns: {len(df.columns)}")
print("Columns:", df.columns.tolist())

# Drop unnecessary columns
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

# Ensure the 'diagnosis' column is the target (Convert 'M' -> 1, 'B' -> 0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Select only the mean features for simplicity and consistency with main.py
mean_features = [col for col in df.columns if '_mean' in col]
X = df[mean_features]
y = df["diagnosis"]

# Debugging check
print(f"Final Feature Columns: {X.shape[1]}")
print(X.columns.tolist())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"Training accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")

# Save model and scaler
joblib.dump(model, "app/model/model.pkl")
joblib.dump(scaler, "app/model/scaler.pkl")

# Save feature names for consistency
feature_names = X.columns.tolist()
joblib.dump(feature_names, "app/model/feature_names.pkl")

print("Model training complete! Files saved in app/model/")