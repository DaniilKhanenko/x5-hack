import pandas as pd
import numpy as np
import re
import random
import ast

TRAIN_PATH = "data/train.csv"
KAGGLE_PATH = "data/russian_supermarket_prices.csv"
OUT_PATH = "train_augmented_simple.csv"

N_BRANDS = 5000
N_BRAND_VOL = 800
N_BRAND_PCT = 400
N_O = 1200

MAX_SPANS_PER_ROW = 2

BRAND_COL = "brand"
VOLUMES = ["500 мл", "1 л", "250 г", "2 кг", "80 г", "65 г"]
PERCENTS = ["5%", "7%", "10%", "12%", "15%"]

random.seed(42)

def normalize_phrase(s: str) -> str:
    return " ".join(str(s).strip().split())

def build_spans_from_tokens(tokens, tags):
    sample = " ".join(tokens)
    spans = []
    pos = 0
    for tok, tag in zip(tokens, tags):
        start = pos
        end = pos + len(tok)
        spans.append((start, end, tag))
        pos = end + 1
    return sample, spans

def is_all_O_and_under_limit(annotation_str, max_len):
    try:
        ann = ast.literal_eval(str(annotation_str))
    except Exception:
        return False
    if not isinstance(ann, list) or len(ann) == 0:
        return False
    if len(ann) > max_len:
        return False
    return all(isinstance(x, tuple) and len(x) == 3 and x[2] == "O" for x in ann)

def add_if_ok(rows_list, sample, spans, max_len):
    if 0 < len(spans) <= max_len:
        rows_list.append({"sample": sample, "annotation": str(spans)})

train = pd.read_csv(TRAIN_PATH, sep=";")
kaggle = pd.read_csv(KAGGLE_PATH)

brands = (
    kaggle[BRAND_COL]
    .dropna()
    .astype(str)
    .map(normalize_phrase)
)
brands = [b for b in pd.unique(brands) if b and 1 <= len(b) <= 60]
random.shuffle(brands)

new_rows = []

for b in brands[:N_BRANDS]:
    btoks = b.split()
    tags = ["B-BRAND"] * len(btoks)
    sample, spans = build_spans_from_tokens(btoks, tags)
    add_if_ok(new_rows, sample, spans, MAX_SPANS_PER_ROW)

for b in brands[N_BRANDS : N_BRANDS + N_BRAND_VOL]:
    vol_toks = random.choice(VOLUMES).split()
    btoks = b.split()
    tokens = btoks + vol_toks
    tags = ["B-BRAND"] * len(btoks) + ["B-VOLUME"] * len(vol_toks)
    sample, spans = build_spans_from_tokens(tokens, tags)
    add_if_ok(new_rows, sample, spans, MAX_SPANS_PER_ROW)

for b in brands[N_BRANDS + N_BRAND_VOL : N_BRANDS + N_BRAND_VOL + N_BRAND_PCT]:
    pct_toks = random.choice(PERCENTS).split()
    btoks = b.split()
    tokens = btoks + pct_toks
    tags = ["B-BRAND"] * len(btoks) + ["B-PERCENT"] * len(pct_toks)
    sample, spans = build_spans_from_tokens(tokens, tags)
    add_if_ok(new_rows, sample, spans, MAX_SPANS_PER_ROW)

o_rows = [{"sample": str(r["sample"]), "annotation": str(r["annotation"])}
          for _, r in train.iterrows() if is_all_O_and_under_limit(r["annotation"], MAX_SPANS_PER_ROW)]
random.shuffle(o_rows)
new_rows.extend(o_rows[:N_O])

aug = pd.DataFrame(new_rows, columns=["sample", "annotation"])
final = pd.concat([train[["sample", "annotation"]], aug], ignore_index=True)
final.to_csv(OUT_PATH, sep=";", index=False, encoding="utf-8")

print(f"Готово: {OUT_PATH}, добавлено строк: {len(aug)}")