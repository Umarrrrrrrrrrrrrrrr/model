from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

# 1️ Load your trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("trained_mini_gpt")
tokenizer = GPT2TokenizerFast.from_pretrained("trained_mini_gpt")

# 2️ Put model in evaluation mode
model.eval()

# 3️ Prompt for generation
prompt = input("Enter a prompt: ")

# 4️ Encode input
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# 5️ Generate text
# Adjust max_length, temperature, top_k, top_p for different creativity
outputs = model.generate(
    input_ids,
    max_length=100,      # total output length
    do_sample=True,      # use sampling instead of greedy
    temperature=0.8,     # creativity level
    top_k=50,            # top-k sampling
    top_p=0.95,          # nucleus sampling
    repetition_penalty=1.1,  # reduce repeats
    num_return_sequences=1
)

# 6️⃣ Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- Generated Text ---")
print(generated_text)
