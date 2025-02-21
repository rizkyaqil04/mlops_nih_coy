import os
import re

# Path untuk menyimpan hasil preprocessing
SCRAPED_DATA_PATH = "data/raw/sinta_papers.json"
PREPROCESSED_DATA_PATH = "data/processed/sinta_preprocessed.json"

# Pastikan direktori 'data' ada
os.makedirs("data/processed", exist_ok=True)

def clean_text(text):
    """ Membersihkan teks dari karakter khusus, angka, dan stopwords. """
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    # Inisialisasi stopwords
    stopword_factory = StopWordRemoverFactory()
    stopwords_id = set(stopword_factory.get_stop_words())
    stopwords_en = set(stopwords.words("english"))

    text = text.lower().strip()  # Konversi ke lowercase & hapus spasi di awal/akhir
    text = re.sub(r"\d+", "", text)  # Hapus angka
    text = re.sub(r"\s+", " ", text)  # Hapus spasi berlebih
    text = re.sub(r"[^\w\s]", "", text)  # Hapus tanda baca
    tokens = word_tokenize(text)  # Tokenisasi
    tokens = [word for word in tokens if word not in stopwords_id and word not in stopwords_en]  # Hapus stopwords
    return " ".join(tokens)

def preprocess_papers(papers):
    """ Membersihkan semua teks dalam hasil scraping Sinta dan menghapus duplikat. """
    seen = set()
    cleaned_papers = []

    for paper in papers:
        title = clean_text(paper.get("title", ""))
        description = clean_text(paper.get("description", ""))

        # Buat tuple unik berdasarkan title dan description
        paper_key = (title, description)

        if paper_key not in seen:
            seen.add(paper_key)
            cleaned_papers.append({"title": title, "description": description})

    return cleaned_papers

if __name__ == "__main__":
    import nltk

    # Pastikan stopwords NLTK sudah diunduh
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

    # Cek apakah file hasil scraping tersedia
    if not os.path.exists(SCRAPED_DATA_PATH):
        print(f"⚠️  File '{SCRAPED_DATA_PATH}' tidak ditemukan. Jalankan 'scraping.py' terlebih dahulu.")
    else:
        # Load data dari file hasil scraping
        with open(SCRAPED_DATA_PATH, "r", encoding="utf-8") as f:
            papers = json.load(f)

        # Preprocess data dan hapus duplikat
        cleaned_papers = preprocess_papers(papers)

        # Simpan hasil preprocessing ke file baru
        with open(PREPROCESSED_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(cleaned_papers, f, indent=2, ensure_ascii=False)

        print(f"✅ Preprocessing selesai! {len(cleaned_papers)} data unik disimpan dalam '{PREPROCESSED_DATA_PATH}'")

