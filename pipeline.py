from preprocess import preprocess_book
from retriever import retrieve_relevant_chunks
from generator import load_t5_model, generate_answer


def qa_pipeline(question, book_path):
    """End-to-end QA pipeline."""

    chunks, bm25_index = preprocess_book(book_path)

    relevant_chunks = retrieve_relevant_chunks(question, bm25_index, chunks)


    model, tokenizer = load_t5_model()


    context = relevant_chunks[0]
    answer = generate_answer(question, context, model, tokenizer)
    return answer


if __name__ == "__main__":

    question = "What did Alice think to herself?"
    book_path = "sample_book.txt"


    answer = qa_pipeline(question, book_path)

    print("Question:", question)
    print("Answer:", answer)