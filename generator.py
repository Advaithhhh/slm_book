from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_t5_model():
    """Load T5-small model and tokenizer."""
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return model, tokenizer

def generate_answer(question, context, model, tokenizer):
    """Generate an answer using T5-small."""
    input_text = f"question: {question} context: {context} answer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Example usage
    model, tokenizer = load_t5_model()
    question = "What did Alice find in the bottle?"
    context = "Alice found a small bottle labeled 'DRINK ME'."
    answer = generate_answer(question, context, model, tokenizer)
    print("Generated Answer:", answer)