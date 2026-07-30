"""
Stream OpenWebText, sentence-split each document, run the annotation model, and
write one annotated JSON record per line to `flansmall_annotated.jsonl`.

Run:
    python bulk_annotation/annotate_pipeline.py
"""
import json
import os
import queue
import threading

import spacy
import ctranslate2
from datasets import load_dataset
from transformers import AutoTokenizer

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
checkpoint = "/home/brianouyang/Post_37.1/text-annotation/text15_checkpoint-28965"           # HF checkpoint (source of the tokenizer)
ct2_model = "/home/brianouyang/Post_37.1/text-annotation/text15_checkpoint-28965-ct2"        # CTranslate2-converted model directory
# Match the checkpoint's training config: it was fine-tuned with
# --max-input-length 128 and --max-target-length 256. Feeding longer inputs is
# out-of-distribution, and a larger decode cap just lets greedy generation run
# longer than the model was ever trained to.
max_input_length = 128       # tokenizer truncation
max_target_length = 256      # CT2 max_decoding_length
prompt = "annotate: "
sent_chunk_size = 992        # sentences batched together per model call
max_batch_tokens = 131072    # CT2 token budget per on-GPU sub-batch
spacy_batch_size = 500       # texts per spaCy batch
spacy_n_process = 8          # leave headroom on the 10-core i9-10900KF
compute_type = "bfloat16"    # CT2 execution precision
prefetch_batches = 8         # how far the CPU stage may run ahead of the GPU

# One JSON object per line -- {"text": "<annotated document>"} -- mirroring the
# schema HuggingFace exposes for Skylion007/openwebtext.
annot_path = "flansmall_annotated.jsonl"


def stream_texts():
    dataset = load_dataset(
        "Skylion007/openwebtext", streaming=True, split="train"
    )
    for example in dataset:
        yield example["text"]


def document_stream(nlp):
    """Yield one document at a time as (original_text, sentences), where
    sentences is a list of (normalized_text, start_char, end_char) tuples.

    Keeping the document boundary intact lets the annotated sentences be glued
    back into the original document shape. start_char/end_char are offsets
    into original_text; the whitespace *between* sentences (paragraph breaks,
    newlines) is reconstructed from those gaps, so document structure is
    preserved. Whitespace *inside* a sentence is normalized to single spaces
    because that is all the model ever sees.
    """
    docs = nlp.pipe(
        stream_texts(),
        n_process=spacy_n_process,
        batch_size=spacy_batch_size,
    )
    for doc in docs:
        sents = []
        for sent in doc.sents:
            text = sent.text
            stripped = text.rstrip()
            trail = len(text) - len(stripped)
            end = sent.end_char - trail
            if stripped:
                sents.append((stripped, sent.start_char, end))
        yield doc.text, sents


def count_done(path):
    """How many complete documents a previous run already wrote. Each output
    line is one document, so the line count is the number done. A crash
    mid-write can leave a partial final line (no trailing newline); drop it
    so the resume boundary is clean."""
    if not os.path.exists(path):
        return 0
    n = 0
    offset = 0
    with open(path, "rb") as f:
        for line in f:
            if line.endswith(b"\n"):
                n += 1
                offset += len(line)
            else:
                break                 # partial trailing line -- truncate it off
    with open(path, "r+b") as f:
        f.truncate(offset)
    return n


def doc_chunks(doc_stream, min_sents):
    """Group whole documents into chunks holding at least `min_sents` sentences
    so the GPU stays fed, while never splitting a document across chunks."""
    chunk = []
    n = 0
    for text, sents in doc_stream:
        chunk.append((text, sents))
        n += len(sents)
        if n >= min_sents:
            yield chunk
            chunk, n = [], 0
    if chunk:
        yield chunk


def reassemble(text, sents, annotated):
    """Rebuild a document from its annotated sentences, restoring the original
    inter-sentence whitespace so only the annotation markers differ from the
    source. `sents` are (norm_text, start_char, end_char); `annotated[i]` is the
    annotated form of sentence i."""
    if not sents:
        return text
    parts = [text[:sents[0][1]]]                 # anything before the 1st sentence
    for i, (_, _start, end) in enumerate(sents):
        parts.append(annotated[i])
        if i + 1 < len(sents):
            parts.append(text[end:sents[i + 1][1]])   # original gap to next sent
    parts.append(text[sents[-1][2]:])            # anything after the last sentence
    return "".join(parts)


# ----------------------------------------------------------------------------
# CPU producer thread: spaCy sentence splitting + tokenization run here, ahead
# of the GPU, so the GPU thread only ever waits on the CT2 forward pass, not
# on Python/spaCy work. spaCy itself parallelizes across `spacy_n_process`
# worker processes; this thread just orchestrates and feeds the queue.
# ----------------------------------------------------------------------------
def encode_sentences(sentences, tokenizer):
    """Batch-encode a flat list of sentences and return CT2 source tokens (one
    list of subword-token strings per sentence)."""
    prompts = [prompt + s for s in sentences]
    enc = tokenizer(prompts, max_length=max_input_length, truncation=True)
    return [tokenizer.convert_ids_to_tokens(ids) for ids in enc["input_ids"]]


def producer(out_queue, nlp, tokenizer, skip=0):
    try:
        stream = document_stream(nlp)
        for i, _ in zip(range(skip), stream):
            if i % 2000 == 0:
                print(f"  skipping... {i}/{skip}", end="\r", flush=True)
        for chunk in doc_chunks(stream, sent_chunk_size):
            # Flatten every sentence in the chunk into one list for the GPU;
            # the per-document sentence counts (len(sents)) let us split the
            # results back out for reassembly.
            flat = [s[0] for (_text, sents) in chunk for s in sents]
            tokens = encode_sentences(flat, tokenizer)
            out_queue.put((chunk, tokens))
    finally:
        out_queue.put(None)          # sentinel: tells the GPU loop we're done


# ----------------------------------------------------------------------------
# GPU thread (main thread): consume pre-tokenized batches, annotate with
# CTranslate2, reassemble each document, write one JSONL record per document.
# ----------------------------------------------------------------------------
def main():
    done = count_done(annot_path)
    if done:
        print(f"Resuming: {done} documents already done, skipping ahead.")

    nlp = spacy.load(
        "en_core_web_sm",
        exclude=["tagger", "attribute_ruler", "lemmatizer", "ner"],
    )
    # CTranslate2 handles length-based sub-batching and padding internally and
    # returns results in input order, so a chunk's sentences come back in the
    # same order we flattened them -- no manual sort/restore needed. The
    # tokenizer stays HF (SentencePiece).
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    translator = ctranslate2.Translator(
        ct2_model, device="cuda", compute_type=compute_type
    )

    batch_queue = queue.Queue(maxsize=prefetch_batches)
    t = threading.Thread(
        target=producer, args=(batch_queue, nlp, tokenizer, done), daemon=True
    )
    t.start()

    docs_written = done
 
    with open(annot_path, "a", encoding="utf-8") as f_annot:
        while True:
            item = batch_queue.get()
            if item is None:
                break
            chunk, source_tokens = item

            results = translator.translate_batch(
                source_tokens,
                max_batch_size=max_batch_tokens,
                batch_type="tokens",
                beam_size=1,
                max_decoding_length=max_target_length,
            )

            # Walk the flat results back out per document, in order.
            ri = 0
            for text, sents in chunk:
                annotated = []
                for _ in sents:
                    res = results[ri]
                    ri += 1
                    out_ids = tokenizer.convert_tokens_to_ids(res.hypotheses[0])
                    out = tokenizer.decode(out_ids, skip_special_tokens=True)
                    annotated.append(out)
                doc_out = reassemble(text, sents, annotated)
                f_annot.write(json.dumps({"text": doc_out}, ensure_ascii=False) + "\n")
                f_annot.flush()

                docs_written += 1
                print(f"  {docs_written} documents written", end="\r", flush=True)

    t.join()
    print(f"\nDone. {docs_written} documents written.")


if __name__ == "__main__":
    main()
