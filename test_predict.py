import joblib

def test_model_loads():
    model = joblib.load('models/sentiment_model.pkl')
    assert model is not None

def test_prediction():
    model = joblib.load('models/sentiment_model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    sample_text = "This course was very helpful!"
    vector = vectorizer.transform([sample_text])
    pred = model.predict(vector)
    assert pred[0] in [0, 1]  # Or your sentiment labels
