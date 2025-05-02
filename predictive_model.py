import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("reviews_with_sentiment.csv")

# Print class distribution before
print("Before SMOTE:\n", df['Review Sentiment'].value_counts())

# Encode sentiment labels to numeric
label_encoder = LabelEncoder()
df['Sentiment_encoded'] = label_encoder.fit_transform(df['Review Sentiment'])

X = df['Review Text']
y = df['Sentiment_encoded']

# Vectorize text
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Split before applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Apply SMOTE to training data only (prevent data leakage)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train Logistic Regression model
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Logistic Regression model, vectorizer, and label encoder saved successfully.")
