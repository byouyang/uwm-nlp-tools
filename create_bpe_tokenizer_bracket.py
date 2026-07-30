"""
Create a GPT-2 tokenizer extended with every bracket symbol in `brackets.py`.

Run:
    python create_bpe_tokenizer_bracket.py
"""
from transformers import AutoTokenizer
from brackets import BRACKET_LIST

path = "tokenizer_bracket"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
vocab = tokenizer.get_vocab().keys()
print(len(BRACKET_LIST))
missing = [b for b in BRACKET_LIST if b in vocab]
print(missing)
num_new_tokens = tokenizer.add_special_tokens({"additional_special_tokens": BRACKET_LIST})
tokenizer.save_pretrained(path)
print(f"saved tokenizer to {path}")
print(f"added {num_new_tokens} new tokens; vocab is now {len(tokenizer)}")
