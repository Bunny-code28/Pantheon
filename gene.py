from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=5, temperature=1.0)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

if __name__ == '__main__':
    # Load the trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("./output")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Generate text based on a prompt
    prompt = "Apple Inc. is"
    generated_texts = generate_text(prompt, model, tokenizer)

    for i, text in enumerate(generated_texts):
        print(f"Generated Text {i+1}: {text}")
