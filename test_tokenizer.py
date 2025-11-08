# test_tokenizer.py
from transformers import GPT2TokenizerFast

def test_tokenizer():
    print(" Testing Tokenizer...")
    
    # Load your tokenizer
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer/")
        print(" Tokenizer loaded successfully!")
    except Exception as e:
        print(f" Failed to load tokenizer: {e}")
        return
    
    print(f"📊 Vocabulary size: {tokenizer.vocab_size}")
    
    # Test 1: Basic encoding/decoding
    print("\n" + "="*50)
    print("TEST 1: Basic Encoding/Decoding")
    print("="*50)
    
    test_text = "Hello robot!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    
    # Test 2: Detailed token breakdown
    print("\n" + "="*50)
    print("TEST 2: Token Breakdown")
    print("="*50)
    
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(test_text)
    
    print(f"Text: '{test_text}'")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    
    print("\nToken to ID mapping:")
    for i, (token, token_id) in enumerate(zip(tokens, token_ids[1:-1])):  # Skip special tokens
        print(f"  {i+1}. '{token}' → {token_id}")
    
    # Test 3: Special tokens
    print("\n" + "="*50)
    print("TEST 3: Special Tokens")
    print("="*50)
    
    print(f"BOS (start): <s> = {tokenizer.bos_token_id}")
    print(f"EOS (end): </s> = {tokenizer.eos_token_id}") 
    print(f"PAD: <pad> = {tokenizer.pad_token_id}")
    print(f"UNK: <unk> = {tokenizer.unk_token_id}")
    
    # Test 4: Training data sample
    print("\n" + "="*50)
    print("TEST 4: Training Data Sample")
    print("="*50)
    
    try:
        with open("data/train.txt", "r", encoding='utf-8') as f:
            sample = f.readline().strip()
            print(f"Sample from training data: '{sample}'")
            print(f"Encoded: {tokenizer.encode(sample)}")
            print(f"Tokens: {tokenizer.tokenize(sample)}")
    except Exception as e:
        print(f" Could not read training data: {e}")
    
    print("\n Tokenizer test completed!")

if __name__ == "__main__":
    test_tokenizer()