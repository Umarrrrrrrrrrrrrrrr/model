from datasets import load_dataset
from transformers import GPT2TokenizerFast

#  Load tokenizer from vocab & merges files (not tokenizer.json)
tokenizer = GPT2TokenizerFast(
    vocab_file="tokenizer/vocab.json",
    merges_file="tokenizer/merges.txt",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>"
)

#  Load your dataset
dataset = load_dataset("text", data_files={"train": "data/train.txt"})

#  Tokenize function
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

#  Apply tokenizer
tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

#  Save tokenized dataset
tokenized.save_to_disk("data/tokenized_dataset")
print(" Dataset tokenized and saved successfully!")
