import streamlit as st
import joblib
import re
from nltk.corpus import stopwords

# 1. Sayfa Tasarımı ve Başlık
st.set_page_config(page_title="Spam Mesaj Tespit Sistemi", page_icon="🛡️", layout="centered")
st.title("🛡️ Spam Mesaj Tespit Sistemi")
st.write("Yapay zeka tabanlı modelimizle gelen mesajların güvenli mi yoksa spam mi olduğunu anında analiz edin.")

# 2. Kaydedilen Modeli ve Vectorizer'ı Yüklüyoruz
@st.cache_resource # Sayfa her yenilendiğinde modeli tekrar yükleyip sistemi yavaşlatmasın diye önbelleğe alıyoruz
def load_models():
    model = joblib.load("models/spam_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_models()
except FileNotFoundError:
    st.error("Model dosyaları bulunamadı! Lütfen önce 'train.py' dosyasını çalıştırın.")
    st.stop()

# 3. preprocess.py'da yazdığımız metin temizleme fonksiyonunun aynısı
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    clean_words = [w for w in words if w not in stop_words]
    return " ".join(clean_words)

# 4. Kullanıcı Giriş Alanı
user_input = st.text_area("Analiz edilmesini istediğiniz mesajı buraya yazın:", height=150, placeholder="Örn: WINNER! As a valued network customer you have been selected to receive a £900 prize reward...")

# 5. Analiz Butonu ve Tahmin Motoru
if st.button("Mesajı Analiz Et", type="primary"):
    if user_input.strip() == "":
        st.warning("Lütfen analiz etmek için bir metin girin!")
    else:
        # Gelen metni yapay zekanın anlayacağı formata sokuyoruz
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        
        # Tahmin yapıyoruz
        prediction = model.predict(vectorized)[0]
        
        # Sonucu ekrana şık bir şekilde basıyoruz
        st.subheader("Analiz Sonucu:")
        if prediction == "spam":
            st.error("🚨 DİKKAT: Bu mesaj %100 SPAM (İstenmeyen Mesaj) olarak tespit edilmiştir!")
        else:
            st.success("✅ GÜVENLİ: Bu mesaj normal (Ham) bir mesajdır.")