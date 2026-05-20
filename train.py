import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Veri yukleme ve hazirlik
df = pd.read_csv("data/cleaned_spam.csv")
df['text_cleaned'] = df['text_cleaned'].fillna('')

X = df['text_cleaned']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. TF-IDF ve Naive Bayes Modeli
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 3. Model Sonuclari
y_pred = model.predict(X_test_tfidf)
print(f"Model Başarı Oranı: {accuracy_score(y_test, y_pred):.4f}")
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))

# 4. Modelleri disari aktarma
with open("models/spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Modeller basariyla kaydedildi.")