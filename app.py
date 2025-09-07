# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # Enable CORS
import pickle
import sqlite3
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("reviews.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review TEXT,
            sentiment TEXT,
            confidence REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Function to save review to database
def save_review_to_db(review, sentiment, confidence):
    conn = sqlite3.connect("reviews.db")
    c = conn.cursor()
    c.execute('''
        INSERT INTO reviews (review, sentiment, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (review, sentiment, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        review = data.get("review")

        # 1. Check if empty
        if not review or review.strip() == "":
            return jsonify({
                "sentiment": "Error",
                "confidence": 0,
                "message": "Please enter a valid review."
            })

        # 2. Basic validation - check if review is meaningful
        words = review.split()
        if len(words) < 3 or not any(c.isalpha() for c in review):
            return jsonify({
                "sentiment": "Error",
                "confidence": 0,
                "message": "Please enter a correct restaurant review."
            })

        # 3. Keyword validation (restaurant-related words)
        keywords = [
            "food", "taste", "service", "restaurant", "staff",
            "meal", "dinner", "lunch", "breakfast", "dish",
            "menu", "drinks", "coffee", "waiter", "ambience",
            "atmosphere", "chef", "snacks", "cuisine", "buffet"
        ]

        review_lower = review.lower()
        if not any(word in review_lower for word in keywords):
            return jsonify({
                "sentiment": "Error",
                "confidence": 0,
                "message": "Please enter a correct restaurant review."
            })

        # 4. Transform review and predict
        review_vec = vectorizer.transform([review])
        prediction = model.predict(review_vec)[0]
        proba = model.predict_proba(review_vec).max() * 100

        # 5. Message based on sentiment
        if prediction == "Positive":
            message = "Great! Keep visiting and enjoy more delicious food."
        elif prediction == "Negative":
            message = "Sorry for the bad experience. The restaurant should improve!"
        else:
            message = "It seems like you had an average experience."

        # 6. Save review to database
        save_review_to_db(review, prediction, round(proba, 2))

        return jsonify({
            "sentiment": prediction,
            "confidence": round(proba, 2),
            "message": message
        })

    except Exception as e:
        return jsonify({
            "sentiment": "Error",
            "confidence": 0,
            "message": str(e)
        })

# Endpoint to view all past reviews
@app.route("/history", methods=["GET"])
def get_history():
    conn = sqlite3.connect("reviews.db")
    c = conn.cursor()
    c.execute("SELECT review, sentiment, confidence, timestamp FROM reviews ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    history = [{"review": r[0], "sentiment": r[1], "confidence": r[2], "timestamp": r[3]} for r in rows]
    return jsonify(history)

if __name__ == "__main__":
    app.run(debug=True)
