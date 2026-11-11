"""Embed free-text 'Other Condition' entries with BiomedBERT (real data only).

    python scripts/embed_other_conditions.py --conditions unique_conditions.txt \
        --out other_condition_embeddings.pkl

Each entry is wrapped as "The patient has {text}." and the last hidden layer
is mean-pooled to a 768-d vector. PCA to 10 dimensions happens later, inside
each training fold. Needs torch and transformers.
"""

import argparse
import pickle

import numpy as np
import pandas as pd

from socialcare.config import BIOMEDBERT, EMB_DIM


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", required=True, help="text file, one entry per line")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import BertModel, BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(BIOMEDBERT)
    model = BertModel.from_pretrained(BIOMEDBERT).eval()
    with open(args.conditions) as f:
        conditions = f.read().splitlines()

    vectors = []
    with torch.no_grad():
        for text in conditions:
            if not text.strip():
                vectors.append(np.full(EMB_DIM, np.nan))
                continue
            enc = tokenizer(f"The patient has {text}.", return_tensors="pt", truncation=True)
            vectors.append(model(**enc).last_hidden_state.mean(dim=1).squeeze().numpy())

    with open(args.out, "wb") as f:
        pickle.dump({"conditions": conditions, "embeddings": pd.DataFrame(vectors)}, f)
    print(f"saved {len(conditions)} embeddings to {args.out}")


if __name__ == "__main__":
    main()
