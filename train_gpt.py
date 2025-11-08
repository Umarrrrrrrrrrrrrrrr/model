from transformers import (
    Trainer, TrainingArguments,
    GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os

# 1. Load or train tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer/")

# ✅ FIX: Explicitly set the pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # or use tokenizer.add_special_tokens({'pad_token': '<pad>'})

print(f"Tokenizer pad token: {tokenizer.pad_token}")
print(f"Tokenizer pad token ID: {tokenizer.pad_token_id}")

# 2. Load and prepare data
def load_training_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return {"text": texts}

# Load data
data = load_training_data("data/train.txt")
dataset = Dataset.from_dict(data)

print(f"Loaded {len(dataset)} training examples")

# 3. Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors=None
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

print("Dataset tokenized successfully!")

# 4. Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal LM, not masked LM
)

# 5. Create model config
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=256,
    n_ctx=256,
    n_embd=256,
    n_layer=4,
    n_head=4,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

# 6. Initialize model
model = GPT2LMHeadModel(config)

# 7. Training arguments 
training_args = TrainingArguments(
    output_dir="trained_mini_gpt",
    overwrite_output_dir=True,
    num_train_epochs=10,           # Increased epochs
    per_device_train_batch_size=2,
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    learning_rate=5e-4,
    weight_decay=0.01,
    warmup_steps=100,              # Added warmup
    prediction_loss_only=True,
    remove_unused_columns=False,   # Important fix!
    dataloader_pin_memory=False,
)

# 8. Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# 9. Train!
print("Starting training...")
trainer.train()

# 10. Save model
trainer.save_model("trained_mini_gpt")
tokenizer.save_pretrained("trained_mini_gpt")

print("Training completed successfully!")