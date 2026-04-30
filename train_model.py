import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("twitter_training.csv")
df = df.drop_duplicates(subset=['Text'])
df = df.dropna(subset=['Text'])
df = df[df['Sentiment'] != 'Irrelevant']

# Features
X = df['Text']
Y = df['Sentiment']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Vectorizer + Model
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vector = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vector, y_train)

# Save both
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model & Vectorizer saved!")