from flask import Flask, request, render_template, jsonify
import re
import string
import joblib
import os

app = Flask(__name__)

# Load trained model and vectorizer
if not os.path.exists("fake_news_model.pkl") or not os.path.exists("vectorizer.pkl"):
    raise FileNotFoundError("⚠️ Model or vectorizer file is missing! Run `train_model.py` first.")

vectorizer = joblib.load("vectorizer.pkl")
classifier = joblib.load("fake_news_model.pkl")

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form.get("news_text", "")

    if not news_text.strip():
        return jsonify({"error": "⚠️ Please enter some text!"}), 400

    processed_text = preprocess_text(news_text)
    input_tfidf = vectorizer.transform([processed_text])
    prediction = classifier.predict(input_tfidf)[0]

    result = "✅ Real News" if prediction == 1 else "❌ Fake News"
    color = "#28a745" if prediction == 1 else "#dc3545"

    return jsonify({"result": result, "color": color})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
