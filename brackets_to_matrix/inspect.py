"""
Quick sanity check for `brackets_to_matrix/parser.py` output.

Memmaps <prefix>_ids.bin / <prefix>_matrix.bin (+ <prefix>_meta.json) and prints,
token by token: the token id, the decoded token, and which brackets are open
around it. Also reconstructs the first document's text and reports basic stats.

Usage:  python brackets_to_matrix/inspect.py [output_prefix=out] [num_tokens_to_show=60]
"""

import json
import sys

import numpy as np

from brackets import BRACKETS_DICT
from b_to_m import _tokenizer, EOS_ID

prefix = sys.argv[1] if len(sys.argv) > 1 else "out"
show = int(sys.argv[2]) if len(sys.argv) > 2 else 60

# --- load -----------------------------------------------------------------
with open(prefix + "_meta.json") as f:
    meta = json.load(f)
C = meta["num_brackets"]
ids = np.memmap(prefix + "_ids.bin", dtype=meta["ids_dtype"], mode="r")
packed = np.memmap(
    prefix + "_matrix.bin",
    dtype=np.uint8,
    mode="r",
).reshape(-1, meta["matrix_row_bytes"])


def unpack_rows(rows):
    return np.unpackbits(
        rows,
        axis=1,
        bitorder=meta["matrix_bitorder"],
    )[:, :C].astype(bool, copy=False)

# Column i of the multi-hot vector corresponds to the i-th opener in BRACKETS_DICT.
col_to_open = list(BRACKETS_DICT.keys())

print("meta:", meta)
print(f"ids    : shape={ids.shape} dtype={ids.dtype}")
print(f"matrix : packed_shape={packed.shape} unpacked_shape=({packed.shape[0]}, {C}) dtype=bool")
assert ids.shape[0] == packed.shape[0] == meta["num_tokens"], "ids/matrix/meta length mismatch!"
print(f"documents (from meta): {meta.get('docs_done')}")

# --- bracket-detection sanity --------------------------------------------
sample = unpack_rows(packed[: min(len(packed), 100_000)])
rows_with_brackets = int(sample.any(axis=1).sum())
print(f"tokens with >=1 open bracket (first {len(sample)}): {rows_with_brackets}")
if rows_with_brackets == 0:
    print("  NOTE: all-zero so far. With the placeholder GPT-2 tokenizer the "
          "brackets are NOT atomic special tokens, so none are detected. Swap in "
          "your real tokenizer to see brackets here.")

# --- reconstruct the first document's text (brackets are stripped) --------
end = 0
while end < len(ids) and int(ids[end]) != EOS_ID:
    end += 1
first_doc = _tokenizer.decode([int(x) for x in ids[:end]])
print("\nfirst document text (decoded, brackets not in token stream):")
print(repr(first_doc))

# --- token-by-token view --------------------------------------------------
print(f"\nfirst {min(show, len(ids))} tokens  (idx | id | token | open brackets):")
preview = unpack_rows(packed[: min(show, len(ids))])
for i in range(min(show, len(ids))):
    tid = int(ids[i])
    tok = "<EOS>" if tid == EOS_ID else _tokenizer.convert_ids_to_tokens(tid)
    open_here = "".join(col_to_open[c] for c in np.nonzero(preview[i])[0])
    print(f"{i:6d} | {tid:<6d} | {repr(tok):<18} | {open_here}")
