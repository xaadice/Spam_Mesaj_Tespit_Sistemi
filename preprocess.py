import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK kütüphanesinin temizlik paketlerini Mac'imize indiriyoruz (İlk çalıştırmada gerekir)
nltk.download('stopwords')
nltk.download('punkt')

try:
    df = pd.read_csv("data/spam.csv", encoding='latin-1')
except FileNotFoundError:
    print("Hata: data/spam.csv dosyası bulunamadı!")
    exit()

# Önceki adımda yaptığımız sütun temizliğini uyguluyoruz
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
df = df.rename(columns={'v1': 'label', 'v2': 'text'})

# --- METİN TEMİZLEME FONKSİYONU ---
def clean_text(text):
    # 1. Tüm harfleri küçük harfe çeviriyoruz
    text = text.lower()
    
    # 2. Noktalama işaretlerini ve sayıları temizliyoruz (Sadece harfler kalsın)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Kelimeleri tek tek ayırıyoruz (Tokenization)
    words = text.split()
    
    # 4. İngilizce etkisiz kelimeleri (stopwords) ayıklıyoruz
    stop_words = set(stopwords.words('english'))
    clean_words = [w for w in words if w not in stop_words]
    
    # 5. Temizlenmiş kelimeleri tekrar birleştirip tek bir metin yapıyoruz
    return " ".join(clean_words)

# Fonksiyonumuzu tüm veri setine uyguluyoruz ve yeni bir sütun oluşturuyoruz
print("Metinler temizleniyor, lütfen bekleyin...")
df['clean_text'] = df['text'].apply(clean_text)

# Sonucu görmek için orijinal mesaj ile temizlenmiş mesajı yan yana yazdıralım
print("\n--- Temizlik Sonrası Örnek Mesajlar ---")
print(df[['text', 'clean_text']].head())

# Temizlenmiş veriyi kaybetmemek için yeni bir CSV olarak kaydedelim
df.to_csv("data/cleaned_spam.csv", index=False)
print("\n✔ Başarılı! Temizlenmiş veri seti 'data/cleaned_spam.csv' olarak kaydedildi.")