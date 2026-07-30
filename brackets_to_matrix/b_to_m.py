"""
Turn one bracket-annotated document into token ids plus a bracket-state matrix
for downstream LLM training.

Each document is a string in which spans of text are wrapped by matching Unicode
"bracket" characters (see brackets.py). The tokenizer has every bracket added as
a *special token*, so each bracket is a single, atomic token id. We tokenize the
whole document once and then work purely on the resulting token-id stream. For
every *non-bracket* token we emit:
  1. the token id, and
  2. a multi-hot vector marking which bracket types are open around that token
     (i.e. the openers currently on the stack when the token is read).

This module is the per-document worker. `brackets_to_matrix(text)` processes one
document and appends an EOS token at its end, so concatenating the per-document
outputs (see parser.py) yields one long stream of tokens (and aligned
bracket vectors) separated by EOS tokens.

Brackets are not guaranteed to be well formed (human error: stray closers,
unclosed openers). Two linear passes over the token ids handle this per document:

  Pass 1 - repair: scan left to right with a stack of open-bracket ids.
           * opening bracket        -> push it.
           * closing bracket whose opener id is NOT anywhere on the stack
                                     -> drop this closer (stray).
           * closing bracket whose opener id IS on the stack
                                     -> it's a valid pair; pop down to that
                                        opener, marking every opener above it
                                        (left unclosed) as invalid.
           * openers still on the stack at the end were never closed -> invalid.
           The surviving brackets are therefore perfectly balanced.

  Pass 2 - build: walk the same token-id stream, skipping the invalid brackets
           and maintaining the open-bracket stack, emitting (token id, multi-hot
           vector) for every non-bracket token. Because pass 1 guarantees balance,
           every surviving closer matches the stack top.
"""

import numpy as np
from brackets import BRACKETS_DICT
from transformers import AutoTokenizer

_tokenizer = AutoTokenizer.from_pretrained("tokenizer_bracket")
EOS_ID = _tokenizer.eos_token_id


def encode(text):
    """Text -> list of token ids (no special tokens added by the template)."""
    return _tokenizer.encode(text, add_special_tokens=False)


# --------------------------------------------------------------------------- #
# Bracket tables, keyed by TOKEN ID. Each bracket char in brackets.py is encoded
# to its atomic special-token id, then the passes below run on ids alone. A
# bracket "type" is indexed by its opener id, which is also its multi-hot column.
# --------------------------------------------------------------------------- #
def _bracket_id(ch):
    """Token id of a single bracket char (an atomic special token -> one id)."""
    return encode(ch)[0]


NUM_BRACKETS = len(BRACKETS_DICT)          # 94 bracket types, one column each
OPEN_TO_CLOSE_ID = {}                      # open id  -> close id
CLOSE_TO_OPEN_ID = {}                      # close id -> open id
BRACKET_INDEX = {}                         # open id  -> multi-hot column
for _col, (_open_ch, _close_ch) in enumerate(BRACKETS_DICT.items()):
    _open_id = _bracket_id(_open_ch)
    _close_id = _bracket_id(_close_ch)
    OPEN_TO_CLOSE_ID[_open_id] = _close_id
    CLOSE_TO_OPEN_ID[_close_id] = _open_id
    BRACKET_INDEX[_open_id] = _col


def _find_invalid(token_ids):
    """Pass 1: return the set of token-stream indices whose bracket breaks nesting."""
    stack = []            # (open_id, index), bottom -> top, still open
    present = {}          # open_id -> how many are currently on the stack
    invalid = set()

    for i, tid in enumerate(token_ids):
        if tid in OPEN_TO_CLOSE_ID:                   # opening bracket
            stack.append((tid, i))
            present[tid] = present.get(tid, 0) + 1
        elif tid in CLOSE_TO_OPEN_ID:                 # closing bracket
            open_id = CLOSE_TO_OPEN_ID[tid]
            if present.get(open_id, 0) == 0:
                invalid.add(i)                        # no opener anywhere -> stray
                continue
            # An opener of this type sits somewhere in the stack. Pop down to the
            # nearest one; every opener above it was left unclosed -> invalid.
            while True:
                top_id, top_i = stack.pop()
                present[top_id] -= 1
                if top_id == open_id:
                    break                             # matched pair: both kept
                invalid.add(top_i)
        # else: ordinary content token -> ignore in pass 1

    # Openers never closed by the end of the document.
    for _open_id, top_i in stack:
        invalid.add(top_i)
    return invalid


def _build(token_ids, invalid):
    """Pass 2: emit (out_ids, vectors) over the repaired token stream."""
    open_vec = np.zeros(NUM_BRACKETS, dtype=bool)
    stack = []
    out_ids = []
    vectors = []

    for i, tid in enumerate(token_ids):
        if tid in OPEN_TO_CLOSE_ID:                   # opening bracket
            if i in invalid:
                continue
            stack.append(tid)
            open_vec[BRACKET_INDEX[tid]] = True
        elif tid in CLOSE_TO_OPEN_ID:                 # closing bracket
            if i in invalid:
                continue
            top = stack.pop()                         # matches top (guaranteed by pass 1)
            open_vec[BRACKET_INDEX[top]] = False
        else:                                         # content token -> emit
            out_ids.append(tid)
            vectors.append(open_vec.copy())

    return out_ids, vectors


def brackets_to_matrix(text):
    """Process ONE document (its text) -> (token_ids, vectors), EOS-terminated.

    token_ids : list[int]        length N (+1 for the trailing EOS)
    vectors   : list[np.ndarray] each shape (NUM_BRACKETS,) bool, aligned to token_ids

    A repaired document is balanced, so the EOS token carries an all-zero vector.
    """
    token_ids = encode(text)                # whole doc; brackets are atomic ids
    invalid = _find_invalid(token_ids)      # pass 1
    out_ids, vectors = _build(token_ids, invalid)   # pass 2

    out_ids.append(EOS_ID)
    vectors.append(np.zeros(NUM_BRACKETS, dtype=bool))
    return out_ids, vectors
