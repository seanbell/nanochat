"""
Common utilities for nanochat.
"""

import os
import re
import logging
import urllib.request
import torch
import torch.distributed as dist
from filelock import FileLock

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def curated_paths(output_name):
    """Return paths dict for all checkpoint files of a curation run."""
    curated_dir = os.path.join(get_base_dir(), "curated")
    return {
        "dir": curated_dir,
        "jsonl": os.path.join(curated_dir, f"{output_name}.jsonl"),
        "embeddings": os.path.join(curated_dir, f"{output_name}_embeddings.npy"),
        "clusters": os.path.join(curated_dir, f"{output_name}_clusters.npy"),
        "ifd": os.path.join(curated_dir, f"{output_name}_ifd.npy"),
        "tokens": os.path.join(curated_dir, f"{output_name}_tokens.npz"),
    }


def load_curated_data(output_name, dataset=None, downsample=0, seed=42):
    """Load curated conversations with cluster/IFD metadata from checkpoint files.

    Handles all loading strategies:
    - JSONL with embedded metadata (fastest path)
    - JSONL without metadata + source dataset for backfill
    - Source dataset with numpy overlays (pre-JSONL browsing)
    - Stats-only from numpy arrays (no conversations)

    Returns dict: conversations, n_total, has_jsonl, completed_stages.
    Each conversation dict will have 'cluster' and 'ifd_diff' keys injected
    when the corresponding numpy checkpoints exist.
    """
    import json
    import time
    import numpy as np

    paths = curated_paths(output_name)
    conversations = []
    n_total = 0
    has_jsonl = False
    completed_stages = []

    # Load numpy checkpoints (these index the full filtered dataset)
    cluster_labels = None
    ifd_diffs = None
    if os.path.exists(paths["clusters"]):
        cluster_labels = np.load(paths["clusters"])
        n_total = len(cluster_labels)
        completed_stages.append("clusters")
        print(f"Loaded clusters: {len(cluster_labels):,} items, {len(np.unique(cluster_labels))} clusters")
    if os.path.exists(paths["ifd"]):
        ifd_diffs = np.load(paths["ifd"])
        n_total = max(n_total, len(ifd_diffs))
        completed_stages.append("ifd")
        print(f"Loaded IFD diffs: {len(ifd_diffs):,} items")
    if os.path.exists(paths["embeddings"]):
        completed_stages.append("embeddings")

    # Load conversations: prefer JSONL, fall back to source dataset
    if os.path.exists(paths["jsonl"]):
        print(f"Loading {paths['jsonl']}...")
        t0 = time.time()
        with open(paths["jsonl"]) as f:
            conversations = [json.loads(line) for line in f]
        has_jsonl = True
        completed_stages.append("jsonl")
        print(f"Loaded {len(conversations):,} curated conversations in {time.time() - t0:.1f}s")

        # Backfill metadata if JSONL was written by older version without it
        if conversations and "cluster" not in conversations[0] and dataset and cluster_labels is not None:
            print("JSONL lacks cluster/ifd metadata. Loading source dataset to backfill...")
            t0 = time.time()
            from tasks.hfchat import load_sft_conversations
            source = load_sft_conversations(dataset, downsample=downsample, seed=seed)
            source_lookup = {}
            for i, conv in enumerate(source):
                msgs = conv.get("messages", [])
                if msgs:
                    source_lookup[msgs[0]["content"]] = i
            mapped = 0
            for conv in conversations:
                msgs = conv.get("messages", [])
                src_idx = source_lookup.get(msgs[0]["content"]) if msgs else None
                if src_idx is not None:
                    if src_idx < len(cluster_labels):
                        conv["cluster"] = int(cluster_labels[src_idx])
                    if ifd_diffs is not None and src_idx < len(ifd_diffs):
                        conv["ifd_diff"] = float(ifd_diffs[src_idx])
                    mapped += 1
            del source, source_lookup
            print(f"Backfilled {mapped:,}/{len(conversations):,} conversations in {time.time() - t0:.1f}s")

    elif dataset:
        print(f"JSONL not yet produced. Loading source dataset: {dataset}")
        t0 = time.time()
        from tasks.hfchat import load_sft_conversations
        conversations = load_sft_conversations(dataset, downsample=downsample, seed=seed)
        print(f"Loaded {len(conversations):,} conversations in {time.time() - t0:.1f}s")

        # Overlay numpy arrays directly (they index the same filtered dataset)
        if cluster_labels is not None:
            if len(cluster_labels) == len(conversations):
                for i, conv in enumerate(conversations):
                    conv["cluster"] = int(cluster_labels[i])
            else:
                print(f"WARNING: cluster array size ({len(cluster_labels)}) != conversation count ({len(conversations)})")
        if ifd_diffs is not None:
            if len(ifd_diffs) == len(conversations):
                for i, conv in enumerate(conversations):
                    conv["ifd_diff"] = float(ifd_diffs[i])
            else:
                print(f"WARNING: IFD array size ({len(ifd_diffs)}) != conversation count ({len(conversations)})")
    else:
        if n_total == 0:
            raise SystemExit(f"No data found for '{output_name}'. Run curate_sft.py first, or pass --dataset.")
        print(f"Stats-only mode: {n_total:,} items from numpy arrays (pass --dataset to browse)")

    return {
        "conversations": conversations,
        "n_total": n_total,
        "has_jsonl": has_jsonl,
        "completed_stages": completed_stages,
    }

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read() # bytes

        # Write to local file
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)

def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    Used to decide whether we *should* initialize a PG.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
    """
    True if torch.distributed is available and the process group is initialized.
    Used at cleanup to avoid destroying a non-existent PG.
    """
    return dist.is_available() and dist.is_initialized()

def get_dist_info():
    if is_ddp_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def autodetect_device_type():
    # prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type

def compute_init(device_type="cuda"): # cuda|cpu|mps
    """Basic initialization that we keep doing over and over, so make common."""

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

    # Reproducibility
    # Note that we set the global seeds here, but most of the code uses explicit rng objects.
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)

    # Precision
    if device_type == "cuda":
        torch.backends.fp32_precision = "tf32" # uses tf32 instead of fp32 for matmuls

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type) # mps|cpu

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp_initialized():
        dist.destroy_process_group()

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

# hardcoded BF16 peak flops for various GPUs
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# and PR: https://github.com/karpathy/nanochat/pull/147
def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    # Table order matters: more specific patterns first.
    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere data center
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        # NVIDIA Ada data center
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA accelerators
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # Consumer RTX
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')
