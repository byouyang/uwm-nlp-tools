"""
Fine-tune FLAN-T5-small to convert "control" text into "annotated" text, line by line.

Data:
  TTEXT16BareMod.txt             -> model input  (plain text)
  TTEXT16ModBrackets_clean.txt   -> model target (annotated text)

Line N of the control file is the input for training example N, and line N of the
annotated file is the ground-truth output. The two files must have the same number
of lines.

Important detail about the tokenizer
------------------------------------
The annotated corpus uses many unusual symbols (e.g. ◅ ⟬ 【 】 ⇨ ⇦ « » ↠ …).
FLAN-T5's default SentencePiece vocabulary does not contain these, so they would all be
collapsed to <unk> and the model could never reproduce them. The bracket symbols
used by the annotation scheme are defined in `brackets.py`; this script iterates
over that dictionary and registers every open/close character with the tokenizer
via the HuggingFace `add_tokens` API (then resizes the model's embedding table to
match). This makes the annotation scheme actually learnable.

Usage
-----
    python bulk_annotation/train_t5_annotator.py
    python bulk_annotation/train_t5_annotator.py --epochs 5 --batch-size 8
    python bulk_annotation/train_t5_annotator.py --max-samples 2000

After training, run inference with:
    python bulk_annotation/train_t5_annotator.py --predict "Some control sentence here."
"""

import argparse
import json
import os
from typing import List

from datasets import Dataset
from transformers import (
    AddedToken,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from brackets import BRACKETS_DICT as BRACKETS

CONTROL_FILE = "TTEXT16BareMod.txt"
ANNOTATED_FILE = "TTEXT16ModBrackets_clean.txt"
TASK_PREFIX = "annotate: "  # T5 is trained with task prefixes; gives the model a hint


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def read_lines(path: str) -> List[str]:
    """Read a file as a list of lines with trailing newlines stripped."""
    with open(path, "r", encoding="utf-8") as f:
        # Keep the line as-is (only strip the line break), so internal whitespace
        # and symbols are preserved exactly.
        return [line.rstrip("\n").rstrip("\r") for line in f]


def load_pairs(control_path: str, annotated_path: str, max_samples: int = 0):
    control = read_lines(control_path)
    annotated = read_lines(annotated_path)

    if len(control) != len(annotated):
        raise ValueError(
            f"Line count mismatch: {control_path} has {len(control)} lines, "
            f"{annotated_path} has {len(annotated)} lines. They must match "
            f"because each line is one training example."
        )

    # Drop pairs where either side is empty after stripping.
    pairs = [
        (c, a)
        for c, a in zip(control, annotated)
        if c.strip() and a.strip()
    ]

    if max_samples and max_samples > 0:
        pairs = pairs[:max_samples]

    inputs = [c for c, _ in pairs]
    targets = [a for _, a in pairs]
    return inputs, targets


# --------------------------------------------------------------------------- #
# Tokenizer: teach it the annotation symbols
# --------------------------------------------------------------------------- #
def add_bracket_symbols(tokenizer) -> int:
    """
    Register every bracket character defined in `brackets.py` with the tokenizer.

    We iterate over the open/close pairs in BRACKETS and hand the full set to the
    HuggingFace `add_tokens` API, which only adds tokens not already present in the
    vocabulary and returns how many were actually added.
    """
    # Collect both the opening and closing characters of every pair.
    symbols = set()
    for open_ch, close_ch in BRACKETS.items():
        symbols.add(open_ch)
        symbols.add(close_ch)

    # Sort for deterministic ordering across runs; add_tokens skips duplicates and
    # any symbol the tokenizer already knows, returning the count actually added.
    return tokenizer.add_tokens(sorted(symbols))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args):
    print("Loading data...")
    inputs, targets = load_pairs(CONTROL_FILE, ANNOTATED_FILE, args.max_samples)
    print(f"  {len(inputs)} training pairs")

    print(f"Loading base model '{args.model}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    print("Adding bracket symbols from brackets.py to the tokenizer...")
    n_added = add_bracket_symbols(tokenizer)
    print(f"  added {n_added} new tokens to the vocabulary")
    if n_added:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # Build the HF dataset and split off a small validation set.
    ds = Dataset.from_dict({"input": inputs, "target": targets})
    split = ds.train_test_split(test_size=args.val_fraction, seed=42)

    def preprocess(batch):
        model_inputs = tokenizer(
            [TASK_PREFIX + x for x in batch["input"]],
            max_length=args.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing...")
    tokenized = split.map(
        preprocess,
        batched=True,
        remove_columns=["input", "target"],
        desc="Tokenizing",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=50,
        eval_strategy="epoch",
        # save_strategy="best" writes a checkpoint only when validation loss
        # improves; with save_total_limit=3 that retains the three best models.
        save_strategy="best",
        save_total_limit=3,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Reload the best (lowest eval_loss) checkpoint at the end so the saved
        # model is the best one, not the final epoch's (which may have overfit).
        # This is also what lets EarlyStoppingCallback restore the best weights.
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        bf16=True,
        fp16=False,  # CPU / no-CUDA friendly; set True if training on GPU
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        # Stop early if eval_loss doesn't improve for `patience` consecutive evals
        # (one eval per epoch here). num_train_epochs becomes an upper bound.
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    print("Training...")
    train_result = trainer.train()

    print(f"Saving final model to '{args.output_dir}'...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Write a metrics file summarizing the run: final train/eval metrics, the best
    # checkpoint chosen, and the full per-epoch log history.
    print("Writing metrics file...")
    final_eval = trainer.evaluate()
    metrics = {
        "train": train_result.metrics,
        "eval": final_eval,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "log_history": trainer.state.log_history,
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  metrics written to '{metrics_path}'")
    print("Done.")


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def predict(args):
    model_dir = args.output_dir
    if not os.path.isdir(model_dir):
        raise SystemExit(
            f"No trained model found at '{model_dir}'. Train first with: "
            f"python {os.path.basename(__file__)}"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    enc = tokenizer(
        TASK_PREFIX + args.predict,
        return_tensors="pt",
        max_length=args.max_input_length,
        truncation=True,
    )
    out = model.generate(
        **enc,
        max_length=args.max_target_length,
        num_beams=args.num_beams,
    )
    print(tokenizer.decode(out[0], skip_special_tokens=True))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="google/flan-t5-small", help="Base model name")
    p.add_argument("--output-dir", default="t5-annotator", help="Where to save the model")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--max-input-length", type=int, default=128)
    p.add_argument("--max-target-length", type=int, default=256)
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early-stopping patience: stop if eval_loss doesn't improve for this "
        "many consecutive evals (epochs). num_train_epochs is the upper bound.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit number of training pairs (0 = use all). Useful for a quick test.",
    )
    p.add_argument("--num-beams", type=int, default=4, help="Beams for --predict")
    p.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Run inference on a single control sentence instead of training.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.predict is not None:
        predict(args)
    else:
        train(args)
