from janome.tokenizer import Tokenizer
from rank_bm25 import BM25Okapi
import tiktoken

documents = [
    "RAGの検索精度を上げるには形態素解析が有効です。",
    "新しい素材の形状を分析し、問題を解決する態度が重要だ。",  # 一般名詞のノイズ文書
    "最新フレームワークのLlamaIndexを導入して構築した。",
    "旧来のIndex構造を見直し、Llamaモデルと連携させた。",  # 新語のノイズ文書
    # 上記とは全く関係の無いノイズ文書
    "今日は天気が良いので公園を散歩した。",
    "週末に友人とカフェでコーヒーを飲んだ。",
    "駅前に新しいラーメン屋がオープンした。",
    "来月の旅行に向けてホテルを予約した。",
    "図書館で歴史の本を借りて読んだ。",
    "夕食にカレーライスを作って食べて散歩した。",
]

# BPEによるトークナイザの準備
encode = tiktoken.get_encoding("o200k_base")
tokenize_bpe = lambda text: [encode.decode([tid]) for tid in encode.encode(text)]

# 形態素解析によるトークナイザの準備
janome = Tokenizer()
tokenize_ja = lambda text: [str(t) for t in janome.tokenize(text, wakati=True)]

bm25_bpe = BM25Okapi([tokenize_bpe(doc) for doc in documents])
bm25_ja = BM25Okapi([tokenize_ja(doc) for doc in documents])

def compare_search(query):
    print(f"=== 検索クエリ: 「{query}」 ===")
    print(f"BPE分割     : {tokenize_bpe(query)}")
    print(f"形態素分割  : {tokenize_ja(query)}")

    # スコアの取得
    scores_bpe = bm25_bpe.get_scores(tokenize_bpe(query))
    scores_ja = bm25_ja.get_scores(tokenize_ja(query))

    # 各文書の中でBPEまたは形態素解析によるスコアが0以上の場合に出力
    for i, doc in enumerate(documents):
        if scores_bpe[i] > 0 or scores_ja[i] > 0:
            print(f"  [文書{i+1}] BPE: {scores_bpe[i]:.4f} | 形態素: {scores_ja[i]:.4f} | {doc}")
    print()

# パターン1：一般的な日本語の複合名詞
compare_search("形態素解析")

# パターン2：英語・新語を含むキーワード
compare_search("LlamaIndex")

# パターン2-b：英語・新語を含むキーワードの表記揺れバージョン
compare_search("Llama Index")
