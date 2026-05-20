import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    
    stop_words = set(stopwords.words('english'))
    clean_words = [w for w in words if w not in stop_words]
    
    return " ".join(clean_words)

if __name__ == "__main__":
    df = pd.read_csv("data/spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    
    df['text_cleaned'] = df['text'].apply(clean_text)
    df.to_csv("data/cleaned_spam.csv", index=False)
    print("Veri on isleme tamamlandi!")