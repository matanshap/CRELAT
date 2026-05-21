import torch
from transformers import GPT2Model, GPT2Config, AutoTokenizer

model_name = "bmeyer2025/tiny-gpt-shakespeare"

try:
    print("Trying AutoTokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded!")
except Exception as e:
    print("Tokenizer failed:", e)

try:
    print("\nTrying GPT2Model with from_pretrained...")
    model = GPT2Model.from_pretrained(model_name)
    print("Model loaded!")
except Exception as e:
    print("Model failed:", e)
