# 🛡️ Yapay Zeka Tabanlı Spam Mesaj Tespit Sistemi

Bu proje, gelen SMS ve anlık mesajların güvenli mi (**Ham**) yoksa istenmeyen mi (**Spam**) olduğunu yüksek doğruluk oranıyla analiz eden uçtan uca bir makine öğrenmesi ve veri bilimi projesidir. Veri setinin temizlenmesinden modelin eğitilmesine ve Streamlit ile kullanıcı dostu bir web arayüzüne dönüştürülmesine kadar tüm aşamaları içerir.

---

## 🚀 Proje Özellikleri
* **Gelişmiş Metin Ön İşleme:** NLTK kütüphanesi kullanılarak noktalama işaretlerinin temizlenmesi, küçük harfe dönüştürme ve etkisiz kelimelerin (stopwords) ayıklanması.
* **Doğruluk Oranı (Accuracy):** %96.59 başarı yakalayan optimize edilmiş Naive Bayes Modeli.
* **Modern Web Arayüzü:** Kullanıcıların mesajları anlık test edebileceği dinamik Streamlit arayüzü.
* **Kurumsal Yapı:** Portfolyo standartlarına uygun, modüler Python kod mimarisi.

---

## 📊 Model Performans Sonuçları

Modelimiz, test veri seti üzerinde yapılan değerlendirmelerde oldukça güçlü metrikler elde etmiştir:

| Metrik | Değer |
| :--- | :--- |
| **Doğruluk Oranı (Accuracy)** | **%96.59** |
| **Ham Mesaj (Ham) Hassasiyeti** | %96 |
| **Spam Mesaj (Spam) Hassasiyeti** | %100 |

---

## 📁 Klasör Yapısı

```text
Spam_Mesaj_Tespit_Sistemi/
├── data/
│   ├── spam.csv            # Kaggle'dan alınan ham veri seti
│   └── cleaned_spam.csv    # Temizlenmiş ve işlenmiş veri seti
├── models/
│   ├── spam_model.pkl      # Eğitilmiş Naive Bayes modeli
│   └── vectorizer.pkl      # Metin sayısallaştırma (TF-IDF) nesnesi
├── .gitignore              # GitHub'a yüklenmeyecek klasörler (.venv vb.)
├── app.py                  # Streamlit Web Arayüzü kodları
├── preprocess.py           # Veri temizleme ve ön işleme betiği
├── README.md               # Proje tanıtım ve kapak sayfası (Bu dosya)
├── requirements.txt        # Projenin çalışması için gerekli kütüphaneler
└── train.py                # Model eğitimi ve performans raporlama betiği

## 🎬 Proje Tanıtım Videosu

Projenin kod yapısını, geliştirme aşamalarını ve çalışan Streamlit arayüzünün canlı testini içeren tanıtım videosunu aşağıdan izleyebilirsiniz:

👉 [Spam Mesaj Tespit Sistemi - YouTube Tanıtım Videosu](https://youtu.be/JIVTvq5KYGI)