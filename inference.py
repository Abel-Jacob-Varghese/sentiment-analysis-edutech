import joblib
from textblob import TextBlob

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def get_polarity(text):
    return TextBlob(text).sentiment.polarity

# Ask user for review
review_text = input("Enter a review: ")

# Convert text into model input
review_vector = vectorizer.transform([review_text])

# Predict
prediction = model.predict(review_vector)
print(f"Predicted Sentiment: {prediction[0]}")
