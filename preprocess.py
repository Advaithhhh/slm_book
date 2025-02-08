import nltk
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import os

nltk.data.path.append(os.path.expanduser("sample_book.txt/nltk_data"))

nltk.download('punkt')

def chunk_text(text, chunk_size=512, overlap=50):
    
    words = word_tokenize(text)
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def build_bm25_index(chunks):
    
    tokenized_chunks = [word_tokenize(chunk) for chunk in chunks]
    return BM25Okapi(tokenized_chunks)

def preprocess_book(book_path):
    with open(book_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    bm25_index = build_bm25_index(chunks)
    return chunks, bm25_index

if __name__ == "__main__":
    chunks, bm25_index = preprocess_book("sample_book.txt")
    print(f"Preprocessed {len(chunks)} chunks.")
