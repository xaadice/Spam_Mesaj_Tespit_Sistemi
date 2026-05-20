import pandas as pd
import joblib  # Modeli kaydetmek için kullanacağız
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Temizlenmiş veri setini yüklüyoruz
try:
    df = pd.read_csv("data/cleaned_spam.csv")
except FileNotFoundError:
    print("Hata: data/cleaned_spam.csv bulunamadı! Önce preprocess.py çalıştırılmalı.")
    exit()

# NaN (boş) satırlar kalmışsa onları temizleyelim
df = df.dropna(subset=['clean_text'])

X = df['clean_text']  # Eğitilecek metinler
y = df['label']       # Hedef etiketler (ham/spam)

# 2. Veriyi %80 Eğitim, %20 Test olarak ikiye bölüyoruz
# Doğru kullanımı 'test_size' şeklindedir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 3. Metinleri Sayılara Dökme (TF-IDF Vectorizer)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Yapay Zeka Modelini Oluşturma ve Eğitme (Naive Bayes)
print("Model eğitiliyor, lütfen bekleyin...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Model Performansını Test Etme
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Performans Sonuçları ---")
print(f"Doğruluk Oranı (Accuracy): %{accuracy * 100:.2f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# 6. Eğitilen Modeli ve Vectorizer'ı Sonra Kullanmak İçin Kaydetme
# Önce modelleri koyacağımız bir klasör oluşturalım (kod otomatik yapar)
import os
if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("\n✔ Başarılı! Model ve Vectorizer 'models/' klasörüne kaydedildi.")