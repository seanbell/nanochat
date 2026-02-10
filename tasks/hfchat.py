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

from datasets import load_dataset
from tasks.common import Task

ALLOWED_ROLES = {"user", "assistant", "system"}

def _has_valid_roles(example):
    """Filter function: keep only conversations with standard roles."""
    return all(m["role"] in ALLOWED_ROLES for m in example["messages"])

class HFChat(Task):

    def __init__(self, dataset_name, split, val_fraction=0.03, max_rows=None, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "val"], "split must be train|val"
        self.dataset_name = dataset_name

        # Load the dataset and figure out splits
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

        # Filter out conversations with non-standard roles (e.g. 'environment')
        before = len(raw)
        self.ds = raw.filter(_has_valid_roles, num_proc=4)
        after = len(self.ds)
        if before != after:
            print(f"HFChat({dataset_name}, {split}): filtered {before - after:,} conversations with non-standard roles ({before:,} -> {after:,})")

        # Optionally truncate to max_rows (applied after filtering, on the shuffled data)
        if max_rows is not None and len(self.ds) > max_rows:
            print(f"HFChat({dataset_name}, {split}): truncating {len(self.ds):,} -> {max_rows:,} rows")
            self.ds = self.ds.select(range(max_rows))

        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        assert len(messages) >= 2, f"Conversation has fewer than 2 messages"
        # Handle optional system message at start
        if messages[0]["role"] == "system":
            rest = messages[1:]
        else:
            rest = messages
        assert len(rest) >= 2, f"Conversation has fewer than 2 non-system messages"
        assert rest[0]["role"] == "user", f"First non-system message must be from user, got {rest[0]['role']}"
        # Ensure all content values are strings (not lists/dicts)
        for msg in messages:
            if not isinstance(msg["content"], str):
                msg["content"] = str(msg["content"])
        return {"messages": messages}
