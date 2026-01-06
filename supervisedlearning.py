import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ---------------------------
# Dataset (large-text ready)
# ---------------------------
data = {
    "text": [
        "Win cash prizes now",
        "Limited offer free reward",
        "Hey are we meeting today",
        "Project discussion at 5 PM",
        "Claim lottery money",
        "How is your family doing",
        "Urgent call required now",
        "Dinner tonight?"
    ],
    "label": [1,1,0,0,1,0,1,0]  # 1 = Spam, 0 = Ham
}

df = pd.DataFrame(data)

X = df["text"]
y = df["label"]

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------
# ML Pipeline (IMPORTANT)
# ---------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1,2)
    )),
    ("model", MultinomialNB(alpha=0.5))
])

# ---------------------------
# Training
# ---------------------------
pipeline.fit(X_train, y_train)

# ---------------------------
# Prediction
# ---------------------------
y_pred = pipeline.predict(X_test)

# ---------------------------
# Evaluation
# ---------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nDetailed Report:\n")
print(classification_report(y_test, y_pred))
