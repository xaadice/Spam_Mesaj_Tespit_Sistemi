# 🛡️ Yapay Zeka Tabanlı Spam Mesaj Tespit Sistemi

Metin tabanlı gelen SMS veya e-posta iletilerinin, makine öğrenmesi algoritmaları kullanılarak ham (güvenli) veya spam (zararlı) olup olmadığını anlık olarak analiz eden web tabanlı bir yazılım projesidir.

---

## 🎬 Proje Tanıtım Videosu

Projenin modüler kod yapısını, geliştirme aşamalarını ve çalışan Streamlit web arayüzünün canlı testlerini içeren detaylı anlatım videosunu aşağıdan izleyebilirsiniz:

👉 [Spam Mesaj Tespit Sistemi - YouTube Tanıtım Videosu](https://youtu.be/JlVTvq5KYGI)

---

## 🛠️ Kullanılan Teknolojiler & Algoritmalar
* **Programlama Dili:** Python
* **Makine Öğrenmesi Modeli:** Multinomial Naive Bayes (Sınıflandırma)
* **Metin Sayısallaştırma:** TF-IDF Vectorizer
* **Veri Ön İşleme (NLP):** NLTK (Stopwords Temizliği, RegEx, Küçük Harf Dönüşümü)
* **Kullanıcı Arayüzü:** Streamlit (Web UI)

### 📂 Klasör Yapısı

```text
Spam_Mesaj_Tespit_Sistemi/
├── data/
│   ├── spam.csv            # Kaggle'dan alınan ham veri seti
│   └── cleaned_spam.csv    # Temizlenmiş ve işlenmiş veri seti
├── models/
│   ├── spam_model.pkl      # Eğitilmiş Naive Bayes modeli
│   └── vectorizer.pkl      # Metin sayısallaştırma nesnesi
├── .gitignore              # GitHub'a yüklenmeyecek klasörler
├── app.py                  # Streamlit Web Arayüzü kodları
├── preprocess.py           # Veri temizleme ve ön işleme betiği
├── README.md               # Proje tanıtım ve kapak sayfası (Bu dosya)
├── requirements.txt        # Proje için gerekli kütüphaneler
└── train.py                # Model eğitimi ve performans raporlama betiği