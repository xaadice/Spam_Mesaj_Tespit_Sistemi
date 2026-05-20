import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Sayfa Yapilandirmasi
st.set_page_config(page_title="Spam Tespit Sistemi", page_icon="🛡️", layout="centered")

# Modelleri yukle
with open("models/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    clean_words = [w for w in words if w not in stop_words]
    return " ".join(clean_words)

# Arayüz Tasarimi
st.title("🛡️ Yapay Zeka Tabanlı Spam Mesaj Tespit Sistemi")
st.write("Gelen SMS veya e-posta metinlerinin güvenli olup olmadığını analiz edin.")

user_input = st.text_area("Analiz edilecek mesajı buraya giriniz:", height=150)

if st.button("Mesajı Analiz Et"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        
        st.subheader("Analiz Sonucu:")
        if prediction == "ham":
            st.success("✅ Bu mesaj GÜVENLİDİR.")
        else:
            st.error("🚨 Bu mesaj bir SPAM / DOLANDIRICILIK mesajı olabilir!")
    else:
        st.warning("Lütfen analiz için geçerli bir metin girin.")