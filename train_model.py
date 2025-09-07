# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# 1. Load dataset
dataset = pd.read_csv("Restaurant_Reviews_5000.tsv", delimiter="\t")

# 2. Map 'Liked' to 'Sentiment'
dataset["Sentiment"] = dataset["Liked"].map({1: "Positive", 0: "Negative"})

# 3. Drop any missing rows (just in case)
dataset = dataset.dropna(subset=["Review", "Sentiment"])

# 4. Add 100 diverse Neutral reviews
neutral_reviews = [
    "The food was okay, nothing special.",
    "Service was average, not too bad.",
    "It was just fine, not great, not terrible.",
    "The restaurant is okay for a casual visit.",
    "I felt neutral about the whole experience.",
    "Nothing stood out, but nothing was bad either.",
    "It was an average dining experience.",
    "Food and service were neither good nor bad.",
    "The ambiance was acceptable.",
    "Just an ordinary restaurant visit.",
    "It met my expectations but didn’t impress.",
    "A standard place for a quick meal.",
    "Nothing noteworthy happened during my visit.",
    "It was decent but not memorable.",
    "Neither satisfied nor dissatisfied.",
    "Average quality food and service.",
    "It was okay, nothing extraordinary.",
    "The restaurant experience was moderate.",
    "Food and service were fine, nothing more.",
    "I have no strong feelings about this place.",
    "The experience was typical, nothing special.",
    "I felt indifferent about the visit.",
    "It was a standard experience, nothing unique.",
    "Nothing to complain about, but nothing exciting either.",
    "The service and food were acceptable.",
    "The restaurant was fine for a casual meal.",
    "Average experience, neither good nor bad.",
    "I have no particular opinion on this visit.",
    "It was an unremarkable dining experience.",
    "Nothing exceptional happened during my visit.",
    "It was just another restaurant experience.",
    "I felt neutral throughout the meal.",
    "The place was okay, neither impressive nor terrible.",
    "Food quality was fine, nothing more.",
    "Service was average, nothing noteworthy.",
    "It met basic expectations, no more.",
    "The restaurant was alright, nothing special.",
    "Nothing remarkable about the experience.",
    "The ambiance was fine, nothing more.",
    "It was an average visit to the restaurant.",
    "Food and service were okay, nothing else.",
    "I didn’t have strong feelings about the experience.",
    "The place was satisfactory for a casual meal.",
    "It was a normal, average dining experience.",
    "Nothing stood out during my visit.",
    "I have a neutral opinion about this place.",
    "The restaurant was standard, nothing unusual.",
    "It was okay, just like any other restaurant.",
    "Service and food were acceptable but not great.",
    "The experience was moderate and neutral.",
    "I found the visit neither good nor bad.",
    "The place was fine, nothing extraordinary.",
    "Average quality, typical restaurant experience.",
    "I felt neither happy nor disappointed.",
    "It was a neutral dining experience overall.",
    "Nothing particularly good or bad about it.",
    "The restaurant met expectations but didn’t exceed them.",
    "It was just an ordinary meal, nothing more.",
    "The food and service were okay, nothing special.",
    "I felt indifferent about the restaurant.",
    "The experience was mediocre, nothing exciting.",
    "Nothing stood out positively or negatively.",
    "It was a standard visit, nothing remarkable.",
    "The service and food were typical and average.",
    "It was a casual visit, neither good nor bad.",
    "I felt neutral during my dining experience.",
    "The restaurant was satisfactory but unremarkable.",
    "It was an average experience, nothing exceptional.",
    "Food and service were adequate, nothing more.",
    "Nothing unusual happened during the visit.",
    "The ambiance and food were acceptable.",
    "I didn’t have any strong opinions about it.",
    "It was an ordinary dining experience.",
    "The restaurant met basic expectations.",
    "Nothing noteworthy occurred during my visit.",
    "It was fine, neither good nor bad.",
    "The experience was standard, nothing memorable.",
    "I felt neutral about the whole dining experience.",
    "Service and food were average and typical.",
    "It was just another visit to a restaurant.",
    "Nothing special happened, it was okay.",
    "The restaurant was neither good nor bad.",
    "I had a neutral experience overall.",
    "It met my expectations but was not exciting.",
    "Average visit, nothing outstanding.",
    "Food and service were fine, just average.",
    "It was an unremarkable experience.",
    "I felt neither pleased nor disappointed.",
    "The restaurant was adequate for a casual meal.",
    "It was an ordinary experience, nothing more.",
    "Nothing stood out positively or negatively.",
    "The ambiance and service were acceptable.",
    "I had no strong feelings about the restaurant.",
    "It was a neutral experience, nothing extraordinary.",
    "Food and service met expectations but didn’t impress.",
    "The restaurant was okay, nothing remarkable.",
    "It was a typical visit to a restaurant.",
    "I felt indifferent about the dining experience.",
    "The experience was moderate and standard.",
    "Nothing exciting or disappointing occurred."
]

neutral_df = pd.DataFrame({"Review": neutral_reviews, "Sentiment": ["Neutral"] * len(neutral_reviews)})

# 5. Combine original + neutral
dataset = pd.concat([dataset, neutral_df], ignore_index=True)

# 6. Features and labels
X = dataset["Review"]
y = dataset["Sentiment"].astype(str)

# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Vectorize text
vectorizer = TfidfVectorizer(max_features=2000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 9. Train Logistic Regression with balanced classes
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# 10. Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# 11. Save model and vectorizer
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully with 100 Neutral reviews!")