from flask import Flask, request, render_template, jsonify
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

app = Flask(__name__)

# Load dataset
df = pd.read_csv("fake_news_dataset.csv")

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

df['text'] = df['text'].apply(preprocess_text)

# Train model
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = vectorizer.fit_transform(df['text'])
classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(X_tfidf, df['label'])

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_text']
    processed_text = preprocess_text(news_text)
    input_tfidf = vectorizer.transform([processed_text])
    prediction = classifier.predict(input_tfidf)[0]
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
