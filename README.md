# UWM REU NLP Tools

Small NLP utilities for annotation training, bracket cleanup, dataset conversion, and model testing.

## Setup

Run: `pip install -r requirements.txt` and `python -m spacy download en_core_web_sm`  
Does: Installs the Python packages and spaCy model used by the scripts.

## Files

| File | Run | Does |
| --- | --- | --- |
| `requirements.txt` | `pip install -r requirements.txt` | Lists the Python dependencies for this repo. |
| `create_bpe_tokenizer_bracket.py` | `python create_bpe_tokenizer_bracket.py` | Builds a GPT-2 tokenizer with all project bracket symbols added as special tokens. |
| `brackets.py` | Not meant to be run directly. | Stores the bracket symbol pairs used across the project. |
| `bulk_annotation/train_t5_annotator.py` | `python bulk_annotation/train_t5_annotator.py` or `python bulk_annotation/train_t5_annotator.py --predict "text"` | Trains the FLAN-T5 annotator on paired text files or runs single-text prediction. |
| `bulk_annotation/annotate_pipeline.py` | `python bulk_annotation/annotate_pipeline.py` | Streams OpenWebText, annotates it with the trained model, and writes JSONL output. |
| `bulk_annotation/get_raw_openweb.py` | `python bulk_annotation/get_raw_openweb.py` | Saves the first 50 OpenWebText records to `openwebtext_first_50.jsonl`. |
| `bracket_mismatch/bracket_mismatch.py` | Not meant to be run directly. | Checks one line for bracket mismatch and returns either the valid line or the error bracket. |
| `bracket_mismatch/mismatch_removal.py` | `python bracket_mismatch/mismatch_removal.py input.txt output.txt` | Writes only bracket-valid lines from an input text file to the output file. |
| `bracket_mismatch/mismatch_line_numbers.py` | `python bracket_mismatch/mismatch_line_numbers.py input.txt mismatch_lines.txt` | Writes each mismatched line number plus the bracket character that caused the error. |
| `brackets_to_matrix/b_to_m.py` | Not meant to be run directly. | Converts one annotated document into token ids plus aligned bracket-state vectors. |
| `brackets_to_matrix/parser.py` | `python brackets_to_matrix/parser.py input.jsonl out` | Converts a JSONL dataset into `out_ids.bin`, `out_matrix.bin`, and `out_meta.json`. |
| `brackets_to_matrix/inspect.py` | `python brackets_to_matrix/inspect.py out 60` | Loads the matrix outputs and prints a quick token-and-bracket sanity check. |
| `regex/remove_annotations.py` | `python regex/remove_annotations.py` | Cleans annotated text files by removing headers, markers, and other artifacts. |
| `regex/remove_markers.py` | `python regex/remove_markers.py` | Removes annotation symbols from a text file to prepare cleaner control text. |
| `testing_llm/nanogpt_ckpt_metadata.py` | `python testing_llm/nanogpt_ckpt_metadata.py path/to/ckpt.pt` | Prints metadata and parameter summaries from a nanoGPT checkpoint. |
| `testing_llm/test_bench.py` | `python testing_llm/test_bench.py` | Evaluates a nanoGPT checkpoint with EleutherAI's lm-evaluation-harness. |
