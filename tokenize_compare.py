import tiktoken
from janome.tokenizer import Tokenizer

text = "RAGの精度を高めるには形態素解析が重要です。"

# 1. LLM用のトークナイズ（GPT-4oなどで使われるBPE：o200k_base）
enc = tiktoken.get_encoding("o200k_base")
bpe_tokens = [enc.decode([tid]) for tid in enc.encode(text)]

# 2. 検索用のトークナイズ（形態素解析：Janome）
janome_tokenizer = Tokenizer()
morpheme_tokens = [str(token) for token in janome_tokenizer.tokenize(text, wakati=True)]

print("【元の文章】", text)
print("【LLM用 (BPE)】", bpe_tokens)
print("【検索用 (形態素解析)】", morpheme_tokens)
