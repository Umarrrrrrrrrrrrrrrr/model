from transformers import (
    Trainer, TrainingArguments,
    GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast
)
from datasets import load_from_disk
import os

# Load tokenizer
tokenizer = GPT2TokenizerFast(
    vocab_file="tokenizer/vocab.json",
    merges_file="tokenizer/merges.txt",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>"
)

# Load tokenized dataset
dataset = load_from_disk("data/tokenized_dataset")["train"]

# Add labels for language modeling
def add_labels(batch):
    batch["labels"] = batch["input_ids"].copy()
    return batch

dataset = dataset.map(add_labels)




#  Create a fresh GPT-2 config (tiny model)
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

#  Initialize a brand-new model
model = GPT2LMHeadModel(config)

#  Training settings
args = TrainingArguments(
    output_dir="trained_mini_gpt",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    logging_steps=10,
    save_steps=200,
    learning_rate=5e-4,
    weight_decay=0.01,
    fp16=True,  # works only if your GPU supports half precision
    report_to="none"  # disable W&B logging
)

#  Train!
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)

trainer.train()

#  Save the trained model
os.makedirs("trained_mini_gpt", exist_ok=True)
trainer.save_model("trained_mini_gpt")
tokenizer.save_pretrained("trained_mini_gpt")

print(" Model trained and saved successfully in 'trained_mini_gpt/'")
