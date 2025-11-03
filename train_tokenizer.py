from tokenizers import ByteLevelBPETokenizer
import os

os.makedirs("tokenizer", exist_ok=True)

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=["data/train.txt"], vocab_size=5000, min_frequency=2,
                special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

tokenizer.save_model("tokenizer/")
tokenizer.save("tokenizer/tokenizer.json")

print(" Tokenizer trained successfully and saved in 'tokenizer/'")
