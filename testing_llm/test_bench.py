"""
Evaluate the nanoGPT checkpoint with EleutherAI's lm-evaluation-harness.

pip install lm-eval

nanoGPT's GPT is not a HuggingFace model, so we wrap it in a tiny adapter that
HFLM can drive: it must expose a `.config` and return a `CausalLMOutput` whose
`.logits` cover the *whole* sequence (not just the last position).
"""
import os

import torch
import transformers
from transformers import AutoTokenizer, GPT2Config
from transformers.modeling_outputs import CausalLMOutput

from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

from model import GPTConfig, GPT
from data.annotated_corpus.add_tokens import add_tokens

# -----------------------------------------------------------------------------
out_dir = "out"
ckpt_name = "ckpt.pt"   # same checkpoint sample.py uses
device = "cuda:0"
tasks = ["lambada_openai"]
limit = 200                          # set to None for the full run
batch_size = 8
# -----------------------------------------------------------------------------

# --- load the nanoGPT model, exactly like sample.py ---
ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint["model_args"])
gpt = GPT(gptconf)


# copied from "resume" path of train.py
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."        # left over from torch.compile during training
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
gpt.load_state_dict(state_dict)
gpt.eval().to(device)

# --- tokenizer, exactly like sample.py (GPT-2 + your extra tokens) ---
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# the harness batches variable-length examples, so a pad token is required
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --- adapter: make nanoGPT like a HF AutoModelForCausalLM ---
class NanoGPTForCausalLM(transformers.PreTrainedModel):
    """Bridges nanoGPT's GPT to the interface lm-eval's HFLM expects."""
    config_class = GPT2Config

    def __init__(self, config, gpt):
        super().__init__(config)
        self.gpt = gpt

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # nanoGPT only returns full-sequence logits when `targets` is given,
        # so pass input_ids as dummy targets to take that code path and
        # discard the loss it computes. logits: (batch, seq, vocab)
        logits, _ = self.gpt(input_ids, targets=input_ids)
        return CausalLMOutput(logits=logits)


hf_config = GPT2Config(
    vocab_size=gptconf.vocab_size,
    n_positions=gptconf.block_size,
    n_ctx=gptconf.block_size,
    n_embd=gptconf.n_embd,
    n_layer=gptconf.n_layer,
    n_head=gptconf.n_head,
)
wrapped = NanoGPTForCausalLM(hf_config, gpt).to(device).eval()

lm_obj = HFLM(
    pretrained=wrapped,                 # pass the object directly, not a path
    tokenizer=tokenizer,
    backend="causal",
    max_length=gptconf.block_size,      # nanoGPT asserts seq_len <= block_size
    batch_size=batch_size,
    device=device,
)

results = evaluator.simple_evaluate(
    model=lm_obj,
    tasks=tasks,
    num_fewshot=0,
    limit=limit,
)

print(results["results"])
