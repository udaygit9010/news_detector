from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("fake_news_model.pkl")  # Ensure correct path
vectorizer = joblib.load("vectorizer.pkl")  # Ensure correct path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form.get("news_text")  # Get input

    if not news_text or news_text.strip() == "":
        return jsonify({'error': 'No text provided'}), 400

    # Vectorize input
    transformed_text = vectorizer.transform([news_text])
    prediction = model.predict(transformed_text)[0]

    # Format result
    result = "Real News ✅" if prediction == 1 else "Fake News ❌"
    color = "#28a745" if prediction == 1 else "#dc3545"

    return jsonify({'result': result, 'color': color})  # Send JSON response

if __name__ == '__main__':
    app.run(debug=True)
