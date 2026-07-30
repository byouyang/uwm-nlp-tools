"""
Parallel driver: convert a JSONL dataset of bracket-annotated documents into one
concatenated (token id, bracket matrix) stream for LLM training.

Pipeline:
  * producer  - streams the dataset document by document with
                datasets.load_dataset(..., streaming=True) and feeds
                (index, text) onto a bounded work queue. On resume it fast-forwards
                with dataset.skip(docs_done) so already-processed docs are re-read.
  * workers   - each pulls a document and runs brackets_to_matrix on it (tokenize,
                repair brackets, build the multi-hot matrix), returning that
                document's (ids, matrix).
  * collector - buffers worker results by index, stitches them back together in the
                original file order, and streams each finished document straight to
                disk (append-only binary). Writes:
                    <prefix>_ids.bin    flat int32  token ids
                    <prefix>_matrix.bin packed bits for each token's bracket vector,
                                         row-major after unpacking
                    <prefix>_meta.json  {docs_done, num_tokens, num_brackets, dtypes}

Every document ends in EOS (added by brackets_to_matrix), so the concatenation is
one long stream separated by EOS tokens. Every input record is treated as one
document (empty text -> just its EOS) so document index == raw record position,
which is what dataset.skip(n) needs for correct resume.

Nothing but the small reorder buffer is held in RAM: the input is streamed
(datasets streaming=True) and the output is appended as documents complete, so a
40 GB (or larger) dataset never needs to fit in memory.

Resume: progress is checkpointed into <prefix>_meta.json every CHECKPOINT_EVERY
documents (bin files are fsync'd first, then meta is replaced atomically). Re-run
the exact same command; if a meta file exists the run truncates the bin files back
to the checkpoint, skips that many input docs, and continues. To force a fresh run,
delete <prefix>_meta.json (and the .bin files).

Read the packed result back lazily with memmap:
    import json, numpy as np
    meta   = json.load(open("out_meta.json"))
    ids    = np.memmap("out_ids.bin",    dtype=meta["ids_dtype"], mode="r")
    packed = np.memmap("out_matrix.bin", dtype=np.uint8, mode="r").reshape(-1, meta["matrix_row_bytes"])
    def unpack_rows(rows):
        return np.unpackbits(rows, axis=1, bitorder=meta["matrix_bitorder"])[:, :meta["num_brackets"]].astype(bool, copy=False)
    batch  = unpack_rows(packed[1000:2000])   # unpack only the rows you need

Usage:  python brackets_to_matrix/parser.py <input.jsonl> <output_prefix>
"""

import json
import os
import sys
import multiprocessing

import numpy as np
from datasets import load_dataset

from b_to_m import brackets_to_matrix, NUM_BRACKETS

NUM_WORKERS = 8
# Bound the work queue so the producer applies backpressure instead of loading
# the whole (streamed) dataset into memory ahead of the workers.
QUEUE_SIZE = NUM_WORKERS * 8
# Token ids are written as int32 (GPT-2 + bracket special tokens fit easily).
ID_DTYPE = np.int32
# Checkpoint progress to the meta file every this many completed documents.
CHECKPOINT_EVERY = 2000
MATRIX_BITORDER = "little"
MATRIX_ROW_BYTES = (NUM_BRACKETS + 7) // 8


def producer(input_path, resource_queue, num_workers, start_index):
    """Stream the dataset and enqueue (index, text), one document at a time.

    start_index resumes past already-processed docs via dataset.skip; index is the
    raw record position so it stays in sync with skip and with the collector.
    """
    dataset = load_dataset("json", data_files=input_path, split="train",
                           streaming=True)
    if start_index:
        dataset = dataset.skip(start_index)
    index = start_index
    for record in dataset:
        text = record.get("text") or ""
        resource_queue.put((index, text))
        index += 1
    # One poison pill per worker so each knows the stream is exhausted.
    for _ in range(num_workers):
        resource_queue.put(None)


def worker(resource_queue, result_queue):
    """Consume documents, emit (index, ids, matrix) for each."""
    while True:
        item = resource_queue.get()
        if item is None:
            break
        index, text = item
        ids, vectors = brackets_to_matrix(text)
        ids_arr = np.asarray(ids, dtype=np.int64)
        matrix = np.stack(vectors)                 # (doc_tokens, NUM_BRACKETS) bool
        result_queue.put((index, ids_arr, matrix))
    # Tell the collector this worker has finished.
    result_queue.put(None)


def _load_checkpoint(output_prefix):
    """Return (docs_done, num_tokens) from a prior run's meta, or (0, 0)."""
    meta_path = output_prefix + "_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return int(meta.get("docs_done", 0)), int(meta.get("num_tokens", 0))
    return 0, 0


def _truncate(path, length):
    """Create path if missing and truncate it DOWN to exactly `length` bytes.

    Drops any bytes written past the last checkpoint (those docs get re-done). It
    only shrinks: if the file is shorter than the checkpoint expects the output is
    inconsistent with the meta, so we fail loudly rather than zero-extend (which
    would inject fake all-zero token rows).
    """
    current = os.path.getsize(path) if os.path.exists(path) else 0
    if current < length:
        raise RuntimeError(
            f"{path} has {current} bytes but the checkpoint expects >= {length}. "
            f"The output files are inconsistent with the meta file; delete the "
            f"_meta.json and .bin files for this prefix to start over.")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        os.ftruncate(fd, length)
    finally:
        os.close(fd)


def _checkpoint(output_prefix, ids_f, matrix_f, docs_done, num_tokens):
    """Durably record progress: fsync the data, then atomically replace the meta."""
    ids_f.flush()
    os.fsync(ids_f.fileno())
    matrix_f.flush()
    os.fsync(matrix_f.fileno())
    meta = {
        "docs_done": docs_done,
        "num_tokens": num_tokens,
        "num_brackets": NUM_BRACKETS,
        "ids_dtype": np.dtype(ID_DTYPE).name,
        "matrix_dtype": "uint8",
        "matrix_encoding": "bitpacked",
        "matrix_bitorder": MATRIX_BITORDER,
        "matrix_row_bytes": MATRIX_ROW_BYTES,
    }
    tmp_path = output_prefix + "_meta.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(meta, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_prefix + "_meta.json")   # atomic


def __main__():
    if len(sys.argv) != 3:
        print("Usage: python build_matrix.py <input.jsonl> <output_prefix>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_prefix = sys.argv[2]
    if output_prefix.endswith(".npy"):
        output_prefix = output_prefix[:-4]

    ids_path = output_prefix + "_ids.bin"
    matrix_path = output_prefix + "_matrix.bin"

    # Resume from the last checkpoint (or start fresh if there is no meta file).
    resume_docs, resume_tokens = _load_checkpoint(output_prefix)
    # Roll the bin files back to exactly what the checkpoint accounts for, so the
    # skipped docs and the on-disk bytes agree even if the last run died mid-chunk.
    _truncate(ids_path, resume_tokens * np.dtype(ID_DTYPE).itemsize)
    _truncate(matrix_path, resume_tokens * MATRIX_ROW_BYTES)
    if resume_docs:
        print(f"Resuming after {resume_docs} docs / {resume_tokens} tokens",
              flush=True)

    resource_queue = multiprocessing.Queue(maxsize=QUEUE_SIZE)
    result_queue = multiprocessing.Queue()

    prod_process = multiprocessing.Process(
        target=producer,
        args=(input_path, resource_queue, NUM_WORKERS, resume_docs),
    )
    worker_processes = [
        multiprocessing.Process(target=worker, args=(resource_queue, result_queue))
        for _ in range(NUM_WORKERS)
    ]

    prod_process.start()
    for w in worker_processes:
        w.start()

    # Workers parse in parallel and may finish out of order, so buffer each result
    # by its document index and write it out only when it is the next document in
    # sequence. Finished documents are streamed straight to disk (append-only), so
    # the only thing in RAM is the small reorder buffer.
    pending = {}
    next_index = resume_docs
    total_tokens = resume_tokens
    last_checkpoint = resume_docs
    workers_done = 0
    with open(ids_path, "ab") as ids_f, open(matrix_path, "ab") as matrix_f:
        while workers_done < NUM_WORKERS:
            item = result_queue.get()
            if item is None:
                workers_done += 1
                continue
            index, ids_arr, matrix = item
            pending[index] = (ids_arr, matrix)
            # Flush every contiguous result to disk, in order, then drop it.
            while next_index in pending:
                ids_i, matrix_i = pending.pop(next_index)
                ids_f.write(ids_i.astype(ID_DTYPE, copy=False).tobytes())
                packed_i = np.packbits(
                    np.ascontiguousarray(matrix_i, dtype=np.uint8),
                    axis=1,
                    bitorder=MATRIX_BITORDER,
                )
                matrix_f.write(packed_i.tobytes())
                total_tokens += int(ids_i.shape[0])
                next_index += 1
                if next_index % 2000 == 0:
                    print(f"Collector: {next_index} documents processed", flush=True)

            # Periodically persist progress so a crash only costs the current chunk.
            if next_index - last_checkpoint >= CHECKPOINT_EVERY:
                _checkpoint(output_prefix, ids_f, matrix_f, next_index, total_tokens)
                last_checkpoint = next_index
                print(f"Checkpoint @ {next_index} docs / {total_tokens} tokens",
                      flush=True)

        # Final checkpoint records the exact completed totals.
        _checkpoint(output_prefix, ids_f, matrix_f, next_index, total_tokens)

    prod_process.join()
    for w in worker_processes:
        w.join()


if __name__ == "__main__":
    __main__()
