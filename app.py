import os
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Get absolute paths for model files
model_path = os.path.join(os.path.dirname(__file__), "fake_news_model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

# Ensure model files exist before loading
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("⚠️ Model or vectorizer file is missing. Upload them before running the app.")

# Load trained model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form.get("news_text")

    if not news_text or news_text.strip() == "":
        return jsonify({'error': '⚠️ Please enter some text!'}), 400

    # Vectorize input
    transformed_text = vectorizer.transform([news_text])
    prediction = model.predict(transformed_text)[0]

    # Format result
    result = "✅ Real News" if prediction == 1 else "❌ Fake News"
    color = "#28a745" if prediction == 1 else "#dc3545"

    return jsonify({'result': result, 'color': color})

if __name__ == '__main__':
    app.run(debug=True)
