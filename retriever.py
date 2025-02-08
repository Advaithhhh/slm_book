from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

def retrieve_relevant_chunks(question, bm25_index, chunks, top_k=3):
    """Retrieve top-k relevant chunks using BM25."""
    tokenized_question = word_tokenize(question)
    scores = bm25_index.get_scores(tokenized_question)
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [chunks[i] for i in top_indices]

if __name__ == "__main__":
    # Example usage
    question = "What did Alice find in the bottle?"
    chunks = ["Chunk 1 text...", "Chunk 2 text..."]  # Replace with actual chunks
    bm25_index = BM25Okapi([word_tokenize(chunk) for chunk in chunks])
    relevant_chunks = retrieve_relevant_chunks(question, bm25_index, chunks)
    print("Relevant Chunks:", relevant_chunks)