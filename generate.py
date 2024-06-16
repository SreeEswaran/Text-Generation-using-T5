import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def generate_text(prompt, max_length=50):
    model_name = "model/t5"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids, 
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        early_stopping=True
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    prompt = "Translate English to French: How are you?"
    generated_text = generate_text(prompt)

    with open("results/generated_texts.txt", "w") as f:
        f.write(generated_text)
    
    print("Generated text saved to results/generated_texts.txt")
