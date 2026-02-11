"""
Generic HuggingFace chat dataset loader.
Loads any HF dataset that has a 'messages' column in standard chat format:
  [{"role": "user"|"assistant"|"system", "content": "..."}]

Usage:
  HFChat("allenai/Dolci-Instruct-SFT", split="train")
  HFChat("allenai/Dolci-Instruct-SFT", split="val", val_fraction=0.03)

If the dataset has a 'test' split, split="val" uses that directly.
Otherwise, the train split is deterministically partitioned using val_fraction.

Conversations with non-standard roles (e.g. 'environment' from tool-calling data)
are automatically filtered out, since the tokenizer expects strict user/assistant alternation.
"""

import copy
import os
from datasets import load_dataset
from nanochat.common import get_base_dir
from tasks.common import Task

# ---------------------------------------------------------------------------
# Chat validation & extraction helpers (used by HFChat and curate_sft)
# ---------------------------------------------------------------------------
def _is_valid_conversation(example):
    """Filter predicate: checks structure, roles, alternation, complete turns."""
    messages = example["messages"]
    if len(messages) < 2:
        return False
    rest = messages[1:] if messages[0]["role"] == "system" else messages
    if len(rest) < 2 or rest[0]["role"] != "user":
        return False
    if len(rest) % 2 != 0:
        return False  # must end with assistant (complete turns only)
    for i, msg in enumerate(rest):
        expected = "user" if i % 2 == 0 else "assistant"
        if msg["role"] != expected:
            return False
    return True

def _stringify_content(example):
    """Map function for datasets.map: ensure all content values are strings."""
    for msg in example["messages"]:
        if not isinstance(msg["content"], str):
            msg["content"] = str(msg["content"])
    return example

def _extract_turns(messages):
    """Extract (user_msg, assistant_msg) pairs from a message list.

    System messages are merged into the first user message (same as tokenizer).
    Returns list of (user_dict, assistant_dict) tuples.
    """
    if messages[0]["role"] == "system":
        messages = copy.deepcopy(messages)
        assert len(messages) > 1 and messages[1]["role"] == "user"
        messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
        messages = messages[1:]
    turns = []
    for i in range(0, len(messages) - 1, 2):
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            turns.append((messages[i], messages[i + 1]))
    return turns

def _count_turns(messages):
    """Count user-assistant turn pairs. No deep copy needed."""
    rest = messages[1:] if messages[0]["role"] == "system" else messages
    return len(rest) // 2

def _conversation_text(messages):
    """Concatenate all message content for embedding."""
    return "\n".join(msg["content"] for msg in messages)

class HFChat(Task):

    def __init__(self, dataset_name, split, val_fraction=0.03, max_rows=None, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "val"], "split must be train|val"
        self.dataset_name = dataset_name

        # Load the dataset — check for local curated JSONL first, then HF Hub
        curated_path = os.path.join(get_base_dir(), "curated", f"{dataset_name}.jsonl")
        if os.path.exists(curated_path):
            ds = load_dataset('json', data_files=curated_path)
        elif os.path.exists(dataset_name):
            ds = load_dataset('json', data_files=dataset_name)
        else:
            ds = load_dataset(dataset_name)
        available_splits = list(ds.keys())

        if split == "val":
            if "test" in available_splits:
                raw = ds["test"]
            elif "validation" in available_splits:
                raw = ds["validation"]
            else:
                # Carve out a validation set from train
                full = ds["train"].shuffle(seed=42)
                n_val = int(len(full) * val_fraction)
                raw = full.select(range(n_val))
        else:
            # split == "train"
            has_val_split = "test" in available_splits or "validation" in available_splits
            full = ds["train"].shuffle(seed=42)
            if has_val_split:
                raw = full
            else:
                # Skip the first val_fraction rows (reserved for val)
                n_val = int(len(full) * val_fraction)
                raw = full.select(range(n_val, len(full)))

        # Filter invalid conversations and stringify content upfront
        before = len(raw)
        raw = raw.filter(_is_valid_conversation, num_proc=4)
        if before != len(raw):
            print(f"HFChat({dataset_name}, {split}): filtered {before - len(raw):,} invalid conversations ({before:,} -> {len(raw):,})")
        raw = raw.map(_stringify_content, num_proc=4)

        # Optionally truncate to max_rows (applied after filtering, on the shuffled data)
        if max_rows is not None and len(raw) > max_rows:
            print(f"HFChat({dataset_name}, {split}): truncating {len(raw):,} -> {max_rows:,} rows")
            raw = raw.select(range(max_rows))

        self.ds = raw
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return {"messages": self.ds[index]["messages"]}
