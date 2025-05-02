import joblib

# Load model and components
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

print("Sentiment Prediction Console (type 'exit' to quit)\n")

while True:
    review = input("Enter a review: ")
    if review.lower() == 'exit':
        print("Exiting sentiment predictor.")
        break

    review_vect = vectorizer.transform([review])
    prediction = model.predict(review_vect)
    sentiment = label_encoder.inverse_transform(prediction)[0]

    print("Predicted Sentiment:", sentiment)
    print("-" * 50)
