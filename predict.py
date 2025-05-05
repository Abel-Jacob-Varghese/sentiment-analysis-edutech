import joblib
import sys

# Load model and components
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

print("Sentiment Prediction Console (type 'exit' to quit)\n")

# Use predefined reviews in non-interactive (CI) mode
if not sys.stdin.isatty():
    test_reviews = [
        "This was a fantastic course!",
        "I didn't learn anything new.",
        "Instructors were great, content was up-to-date."
    ]

    for review in test_reviews:
        review_vect = vectorizer.transform([review])
        prediction = model.predict(review_vect)
        sentiment = label_encoder.inverse_transform(prediction)[0]

        print(f"Review: {review}")
        print("Predicted Sentiment:", sentiment)
        print("-" * 50)

else:
    # Regular interactive mode for local testing
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
