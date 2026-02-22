# app.py — AI-Powered University Chatbot (DIT University)
# Authors: Arpit Sharma, Paras Kumar, Vinayak Sharma
# Guide: Ms. Palak Arora

from flask import Flask, render_template, request, jsonify
import json
import random
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download once
nltk.download('punkt')

app = Flask(__name__)

# Load intents
with open("intents.json", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

# Prepare training data
sentences = []
tags = []

for intent in intents:
    for pattern in intent["patterns"]:
        sentences.append(pattern.lower())
        tags.append(intent["tag"])

# Vectorization (TF-IDF for better results)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(sentences)
y = np.array(tags)

# Train model
model = MultinomialNB()
model.fit(X, y)

def get_bot_reply(user_text):
    user_text = user_text.lower()
    X_test = vectorizer.transform([user_text])
    predicted_tag = model.predict(X_test)[0]

    for intent in intents:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn’t catch that 😅. Try asking about admissions, fees, placements, hostels, campus life, scholarships, or faculty."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_response():
    msg = request.form.get("msg")
    reply = get_bot_reply(msg)
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(debug=True)