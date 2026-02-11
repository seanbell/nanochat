"""
SFT Data Curation via Embedding Clustering + IFD Scoring.

Pipeline:
  1. Load & filter HF dataset (reuse HFChat's role validation)
  2. Embed all conversations (Qwen3-Embedding-0.6B)
  3. K-means cluster (k=256) for diversity
  4. IFD score each sample using nanochat pretrained model (per-turn, averaged)
  5. Filter IFD diff >= 0 (pathological samples where instruction hurts)
  6. Sample target_rows total, uniform across clusters, preferring high-IFD within each
  7. Save curated JSONL

References:
  - Superfiltering (Li et al., ACL 2024): IFD scoring methodology
  - ClusterClip (Shao et al., ACL Findings 2024): uniform cluster allocation
  - OLMo 3 (AI2, 2025): quality filtering, ablation-driven mixing

Usage:
  .venv/bin/python -m scripts.curate_sft \
      --dataset allenai/Dolci-Instruct-SFT \
      --output-name dolci_curated_500k \
      --target-rows 500000 \
      --model-tag baseline2_fullsweep_d20
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from typing import NamedTuple

import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm

from nanochat.common import get_base_dir
from tasks.hfchat import (
    _is_valid_conversation, _stringify_content,
    _extract_turns, _count_turns, _conversation_text,
)

NUM_CPU_PROC = min(96, os.cpu_count() or 1)
EMBED_PROMPT = "Instruct: Represent this conversation for similarity clustering\nQuery: "

def _stage(name):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Curate SFT data via embedding clustering + IFD scoring")
parser.add_argument("--dataset", required=True, help="HuggingFace dataset name (e.g. allenai/Dolci-Instruct-SFT)")
parser.add_argument("--output-name", required=True, help="Output name (saved to $NANOCHAT_BASE_DIR/curated/{name}.jsonl)")
parser.add_argument("--model-tag", required=True, help="Nanochat pretrained model tag for IFD scoring")
parser.add_argument("--target-rows", type=int, default=None, help="Number of rows to select (default: 500k, or downsample/2 if --downsample)")
parser.add_argument("--num-clusters", type=int, default=None, help="Number of k-means clusters (default: 256, or downsample/32 if --downsample)")
parser.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B", help="Embedding model")
parser.add_argument("--embed-batch-size", type=int, default=256, help="Embedding batch size (per GPU)")
parser.add_argument("--ifd-batch-size", type=int, default=32, help="IFD forward pass batch size (actual GPU batch is 2x due to cond+uncond pairing)")
parser.add_argument("--min-embed-dist", type=float, default=0.15, help="Min cosine distance between selected samples within a cluster. 0 = no dedup. 0.15 = skip if cosine_sim > 0.85.")
parser.add_argument("--downsample", type=int, default=0, help="Downsample input to N rows after filtering (0 = no downsampling). Auto-scales --num-clusters and --target-rows if they are at defaults.")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

# ---------------------------------------------------------------------------
# Multi-GPU dispatch
# ---------------------------------------------------------------------------
def run_gpu_workers(worker_fn, per_gpu_args, total_batches, desc):
    """Run worker_fn on each GPU. 1 GPU: direct call. N GPUs: spawn + progress poll.

    worker_fn's last parameter must be progress_counter (default None).
    per_gpu_args: list of arg-tuples, one per GPU.
    """
    if len(per_gpu_args) == 1:
        worker_fn(*per_gpu_args[0])
        return
    counter = mp.Value('i', 0)
    processes = []
    for args in per_gpu_args:
        p = mp.Process(target=worker_fn, args=(*args, counter))
        p.start()
        processes.append(p)
    # Poll shared counter, display tqdm bar, kill children on Ctrl+C
    try:
        with tqdm(total=total_batches, desc=desc, unit="batch") as pbar:
            while any(p.is_alive() for p in processes):
                pbar.update(counter.value - pbar.n)
                time.sleep(0.5)
            pbar.update(counter.value - pbar.n)
        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker {p.pid} exited with code {p.exitcode}")
    except BaseException:
        for p in processes:
            if p.is_alive():
                p.kill()
        for p in processes:
            p.join(timeout=5)
        raise


def embed_worker(gpu_id, shard_ids, sorted_idx, batch_size, pad_token_id,
                 embed_model_name, output_path, progress_counter=None):
    """Batch, pad, and run embedding forward passes on a single GPU.

    Receives raw token ID lists (plain Python) instead of pre-batched tensors
    so that mp.spawn doesn't create one shared-memory FD per tensor.
    """
    if not shard_ids:
        np.save(output_path, np.empty((0, 0)))
        return
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import batch_to_device

    # Dynamic batching: token budget keeps GPU memory ~constant across sequence lengths.
    # Build lazily (generator) so we don't hold all padded tensors in CPU RAM at once.
    max_tokens = batch_size * 512

    num_batches, i = 0, 0
    while i < len(sorted_idx):
        bs = max(1, min(batch_size, max_tokens // len(shard_ids[sorted_idx[i]])))
        i += min(bs, len(sorted_idx) - i)
        num_batches += 1

    def _make_batches():
        i = 0
        while i < len(sorted_idx):
            bs = max(1, min(batch_size, max_tokens // len(shard_ids[sorted_idx[i]])))
            end = min(i + bs, len(sorted_idx))
            batch_ids = [shard_ids[j] for j in sorted_idx[i:end]]
            seq_len = max(len(ids) for ids in batch_ids)
            b = len(batch_ids)
            padded = torch.full((b, seq_len), pad_token_id, dtype=torch.long)
            attn = torch.zeros(b, seq_len, dtype=torch.long)
            for k, ids in enumerate(batch_ids):
                padded[k, seq_len - len(ids):] = torch.tensor(ids, dtype=torch.long)  # left-padded
                attn[k, seq_len - len(ids):] = 1
            yield {"input_ids": padded, "attention_mask": attn}
            i = end

    batches = _make_batches()

    # As recommended in https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
    model = SentenceTransformer(
        embed_model_name, device=device,
        model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
        tokenizer_kwargs={"padding_side": "left"},
    )

    prompt_length = model._get_prompt_length(EMBED_PROMPT)
    extra_features = {"prompt_length": prompt_length} if prompt_length is not None else {}

    all_embeddings = []
    model.eval()
    with torch.no_grad():
        it = batches if progress_counter is not None else tqdm(
            batches, total=num_batches, desc=f"GPU {gpu_id}", unit="batch")
        for features in it:
            features = batch_to_device(features, device)
            features.update(extra_features)
            out = model.forward(features)
            all_embeddings.append(out["sentence_embedding"].detach().float().cpu().numpy())
            if progress_counter is not None:
                with progress_counter.get_lock():
                    progress_counter.value += 1

    # Restore original order
    embeddings = np.empty((len(sorted_idx), all_embeddings[0].shape[1]), dtype=np.float32)
    embeddings[sorted_idx] = np.concatenate(all_embeddings, axis=0)

    np.save(output_path, embeddings)

    del model
    gc.collect()
    torch.cuda.empty_cache()


class TurnItem(NamedTuple):
    """A single turn-pair prepared for IFD scoring."""
    conv_idx: int
    ids_cond: list
    mask_cond: list
    ids_uncond: list
    mask_uncond: list


class IFDDataset(torch.utils.data.Dataset):
    """Dataset for IFD scoring — each item is one turn-pair.

    Pre-extracts turns in __init__, tokenizes on-the-fly in DataLoader workers.
    Must be at module level so spawned workers can import it.
    """

    def __init__(self, conv_shard):
        self.items = []  # flat list of (conv_idx, user_msg, asst_msg)
        for conv_idx, conv in conv_shard:
            for user_msg, asst_msg in _extract_turns(conv["messages"]):
                self.items.append((conv_idx, user_msg, asst_msg))
        self._tokenizer = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self._tokenizer is None:
            from nanochat.tokenizer import get_tokenizer
            self._tokenizer = get_tokenizer()

        conv_idx, user_msg, asst_msg = self.items[idx]

        ids_cond, mask_cond = self._tokenizer.render_conversation(
            {"messages": [user_msg, asst_msg]}
        )
        ids_uncond, mask_uncond = self._tokenizer.render_conversation(
            {"messages": [{"role": "user", "content": ""}, asst_msg]}
        )
        # Skip turns where response was fully truncated (no supervised tokens)
        if sum(mask_cond) == 0 or sum(mask_uncond) == 0:
            return None
        return TurnItem(conv_idx, ids_cond, mask_cond, ids_uncond, mask_uncond)


def ifd_collate_fn(batch):
    """Collate TurnItems into padded tensors for GPU inference.

    Filters None sentinels (truncated turns), pads variable-length cond+uncond
    pairs. Returns None if entire batch was truncated.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    cond_items = [(item.ids_cond, item.mask_cond) for item in batch]
    uncond_items = [(item.ids_uncond, item.mask_uncond) for item in batch]
    all_items = cond_items + uncond_items

    max_len = max(len(ids) for ids, mask in all_items)
    B_total = len(all_items)
    input_ids = torch.full((B_total, max_len - 1), 0, dtype=torch.long)
    target_ids = torch.full((B_total, max_len - 1), -1, dtype=torch.long)
    response_mask = torch.zeros((B_total, max_len - 1), dtype=torch.float32)

    for i, (ids, mask) in enumerate(all_items):
        seq_len = len(ids)
        ids_t = torch.tensor(ids, dtype=torch.long)
        mask_t = torch.tensor(mask, dtype=torch.long)
        input_ids[i, :seq_len - 1] = ids_t[:-1]
        target_ids[i, :seq_len - 1] = ids_t[1:]
        response_mask[i, :seq_len - 1] = mask_t[1:].float()
        target_ids[i, :seq_len - 1][mask_t[1:] == 0] = -1

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "response_mask": response_mask,
        "conv_indices": [item.conv_idx for item in batch],
        "n_pairs": len(batch),
    }


def ifd_worker(gpu_id, conv_shard, model_tag, ifd_batch_size, num_tok_workers,
               result_dict, progress_counter=None):
    """Score a shard of conversations on a single GPU using DataLoader.

    DataLoader workers tokenize turn-pairs in parallel on CPU while the GPU
    runs inference on previously tokenized batches.

    Args:
        conv_shard: list of (global_index, conversation) tuples for this GPU.
        ifd_batch_size: number of turn-pairs per batch.
        num_tok_workers: DataLoader workers for parallel tokenization.
        progress_counter: if provided (mp.Value), increment per batch instead of tqdm.
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    from nanochat.checkpoint_manager import load_model
    model, _, _ = load_model("base", device, phase="eval", model_tag=model_tag)

    dataloader = torch.utils.data.DataLoader(
        IFDDataset(conv_shard),
        batch_size=ifd_batch_size,
        num_workers=num_tok_workers,
        collate_fn=ifd_collate_fn,
        persistent_workers=(num_tok_workers > 0),
        shuffle=False,
        pin_memory=True,
    )

    turn_diffs = defaultdict(list)  # conv_idx -> list of diffs
    skipped_batches = 0

    it = dataloader if progress_counter is not None else (
        tqdm(dataloader, desc=f"GPU {gpu_id} IFD", unit="batch"))

    for batch_dict in it:
        if batch_dict is None:
            skipped_batches += 1
            if progress_counter is not None:
                with progress_counter.get_lock():
                    progress_counter.value += 1
            continue

        input_ids = batch_dict["input_ids"].to(device)
        target_ids = batch_dict["target_ids"].to(device)
        response_mask = batch_dict["response_mask"].to(device)
        conv_indices = batch_dict["conv_indices"]
        n = batch_dict["n_pairs"]

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss_flat = model(input_ids, target_ids, loss_reduction="none")
            loss_2d = loss_flat.float().view(input_ids.size(0), -1)

        mask_sum = response_mask.sum(dim=1)
        avg_losses = (loss_2d.sum(dim=1) / mask_sum).cpu().tolist()

        cond_losses = avg_losses[:n]
        uncond_losses = avg_losses[n:]
        for i in range(n):
            turn_diffs[conv_indices[i]].append(cond_losses[i] - uncond_losses[i])

        if progress_counter is not None:
            with progress_counter.get_lock():
                progress_counter.value += 1

    if skipped_batches > 0:
        print(f"GPU {gpu_id}: skipped {skipped_batches} empty batches (all turns truncated)")

    # Average per-turn diffs to get per-conversation IFD
    ifd_scores = {
        conv_idx: np.mean(turn_diffs[conv_idx]) if conv_idx in turn_diffs else 0.0
        for conv_idx, _ in conv_shard
    }

    result_dict[gpu_id] = ifd_scores


# ---------------------------------------------------------------------------
# Main pipeline (guarded for multiprocessing spawn safety)
# ---------------------------------------------------------------------------
def main():
    args = parser.parse_args()
    # Reduce CUDA memory fragmentation so reserved-but-unallocated blocks stay usable
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    mp.set_start_method("spawn", force=True)

    _stage("Stage 1: Load & Filter")
    t0 = time.time()

    ds = load_dataset(args.dataset)
    raw = ds["train"]
    print(f"Loaded {len(raw):,} rows from {args.dataset}")

    # Downsample early (before filtering) to speed up the entire pipeline
    if args.downsample > 0 and args.downsample < len(raw):
        rng = np.random.RandomState(args.seed)
        keep = rng.choice(len(raw), size=args.downsample, replace=False)
        keep.sort()
        raw = raw.select(keep)
        print(f"Downsampled to {len(raw):,} rows (--downsample {args.downsample})")

    # Fill in defaults (after downsample so we can scale to dataset size)
    if args.num_clusters is None:
        args.num_clusters = max(4, min(256, len(raw) // 32))
        print(f"  Auto --num-clusters={args.num_clusters}")
    if args.target_rows is None:
        args.target_rows = max(10, min(500_000, len(raw) // 2))
        print(f"  Auto --target-rows={args.target_rows}")

    # Filter invalid conversations + stringify content (multi-process)
    before = len(raw)
    raw = raw.filter(_is_valid_conversation, num_proc=NUM_CPU_PROC, desc="Validating conversations")
    raw = raw.map(_stringify_content, num_proc=NUM_CPU_PROC, desc="Stringify content")
    if before != len(raw):
        print(f"Filtered {before - len(raw):,} invalid conversations ({before:,} -> {len(raw):,})")

    conversations = []
    for batch in tqdm(raw.iter(batch_size=10000), total=(len(raw) + 9999) // 10000, desc="Extracting conversations"):
        conversations.extend({"messages": msgs} for msgs in batch["messages"])
    print(f"Stage 1 done: {len(conversations):,} conversations in {time.time() - t0:.1f}s")

    # ----------
    _stage("Stage 2: Embed")

    base_dir = get_base_dir()
    curated_dir = os.path.join(base_dir, "curated")
    os.makedirs(curated_dir, exist_ok=True)
    embed_path = os.path.join(curated_dir, f"{args.output_name}_embeddings.npy")

    if os.path.exists(embed_path):
        print(f"Checkpoint found: {embed_path}, loading...")
        embeddings = np.load(embed_path)
        assert len(embeddings) == len(conversations), \
            f"Embedding count mismatch: {len(embeddings)} != {len(conversations)}. Delete {embed_path} to re-embed."
        print(f"Loaded embeddings: shape {embeddings.shape}")
    else:
        t0 = time.time()
        num_gpus = torch.cuda.device_count()
        print(f"Embedding with {num_gpus} GPU(s), model={args.embed_model}")

        # Tokenize (or load checkpoint)
        tok_path = os.path.join(curated_dir, f"{args.output_name}_tokens.npz")
        if not os.path.exists(tok_path):
            texts = [_conversation_text(conv["messages"]) for conv in tqdm(conversations, desc="Extracting conversation texts")]

            # Print token length distribution (char length ≈ 4 chars/token as rough proxy)
            char_lens = np.array([len(t) for t in texts])
            p10, p25, p50, p75, p90, p99 = np.percentile(char_lens, [10, 25, 50, 75, 90, 99])
            print(f"Text char lengths — min: {char_lens.min():,}, p10: {p10:,.0f}, p25: {p25:,.0f}, "
                  f"p50: {p50:,.0f}, p75: {p75:,.0f}, p90: {p90:,.0f}, p99: {p99:,.0f}, max: {char_lens.max():,}")
            est_trunc_chars = 2048 * 4
            n_truncated = int((char_lens > est_trunc_chars).sum())
            print(f"Estimated truncated at 2048 tokens (~{est_trunc_chars:,} chars): {n_truncated:,} ({100*n_truncated/len(texts):.1f}%)")

            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            from sentence_transformers import SentenceTransformer
            print("Loading tokenizer...")
            st_model = SentenceTransformer(args.embed_model, device="cpu")
            st_model.max_seq_length = 2048

            texts_with_prompt = [EMBED_PROMPT + t for t in texts]
            del texts
            tokenizer = st_model.tokenizer
            all_ids = []
            chunk_size = 50_000
            for start in tqdm(range(0, len(texts_with_prompt), chunk_size),
                              desc="Tokenizing", unit="chunk"):
                chunk = texts_with_prompt[start:start + chunk_size]
                encoded = tokenizer(chunk, padding=False, truncation=True,
                                    max_length=st_model.max_seq_length, return_attention_mask=False)
                all_ids.extend(encoded["input_ids"])
            del texts_with_prompt

            flat = np.array([tok for ids in all_ids for tok in ids], dtype=np.int32)
            lengths = np.array([len(ids) for ids in all_ids], dtype=np.int32)
            np.savez_compressed(tok_path, flat=flat, lengths=lengths, pad_token_id=tokenizer.pad_token_id)
            del st_model, all_ids, flat, lengths
            gc.collect()
            print(f"Saved tokenization checkpoint to {tok_path}")

        tok_data = np.load(tok_path)
        flat, lengths = tok_data["flat"], tok_data["lengths"]
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        all_ids = [flat[offsets[i]:offsets[i+1]].tolist() for i in range(len(lengths))]
        pad_token_id = int(tok_data["pad_token_id"])
        del tok_data, flat, lengths, offsets
        print(f"{len(all_ids):,} tokenized texts")

        # Shard by GPU, sort by length (workers batch & pad locally)
        shard_size = (len(all_ids) + num_gpus - 1) // num_gpus
        shard_paths = []
        gpu_shard_ids = []
        shard_sorted_idx = []

        for gpu_id in range(num_gpus):
            start = gpu_id * shard_size
            end = min(start + shard_size, len(all_ids))
            shard_ids = all_ids[start:end]
            shard_paths.append(os.path.join(curated_dir, f"{args.output_name}_embed_shard_{gpu_id}.npy"))

            # Sort by length to minimize padding waste
            sorted_idx = np.argsort([-len(ids) for ids in shard_ids])
            shard_sorted_idx.append(sorted_idx)
            gpu_shard_ids.append(shard_ids)

        del all_ids
        gc.collect()
        print("Launching GPU workers...")

        max_tokens = args.embed_batch_size * 512
        total_batches = 0
        for gpu_id in range(num_gpus):
            idx = shard_sorted_idx[gpu_id]
            ids = gpu_shard_ids[gpu_id]
            i = 0
            while i < len(idx):
                bs = max(1, min(args.embed_batch_size, max_tokens // len(ids[idx[i]])))
                i += min(bs, len(idx) - i)
                total_batches += 1
        per_gpu_args = [
            (gpu_id, gpu_shard_ids[gpu_id], shard_sorted_idx[gpu_id],
             args.embed_batch_size, pad_token_id,
             args.embed_model, shard_paths[gpu_id])
            for gpu_id in range(num_gpus)
        ]
        run_gpu_workers(embed_worker, per_gpu_args, total_batches, "Embedding")

        # Concatenate shards and save final checkpoint
        embeddings = np.concatenate([np.load(path) for path in shard_paths], axis=0)
        assert len(embeddings) == len(conversations), \
            f"Embedding count mismatch after concat: {len(embeddings)} != {len(conversations)}"
        np.save(embed_path, embeddings)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Saved embeddings to {embed_path}")

        # Clean up shard files
        for path in shard_paths:
            os.remove(path)

        print(f"Stage 2 done in {time.time() - t0:.1f}s")

    # ----------
    _stage("Stage 3: Cluster")

    cluster_path = os.path.join(curated_dir, f"{args.output_name}_clusters.npy")
    if os.path.exists(cluster_path):
        print(f"Checkpoint found: {cluster_path}, loading...")
        cluster_labels = np.load(cluster_path)
        assert len(cluster_labels) == len(conversations), \
            f"Cluster count mismatch: {len(cluster_labels)} != {len(conversations)}. Delete {cluster_path} to re-cluster."
    else:
        t0 = time.time()
        from sklearn.cluster import MiniBatchKMeans

        print(f"Running MiniBatchKMeans with k={args.num_clusters}")
        kmeans = MiniBatchKMeans(
            n_clusters=args.num_clusters, batch_size=16384,
            random_state=args.seed, n_init=10
        )
        cluster_labels = kmeans.fit_predict(embeddings)
        np.save(cluster_path, cluster_labels)
        print(f"Saved cluster labels to {cluster_path}")
        print(f"Stage 3 done in {time.time() - t0:.1f}s")

    # Embeddings kept for dedup in Stage 5; free after selection

    # ----------
    _stage("Stage 4: IFD Score")

    ifd_path = os.path.join(curated_dir, f"{args.output_name}_ifd.npy")
    if os.path.exists(ifd_path):
        print(f"Checkpoint found: {ifd_path}, loading...")
        ifd_diffs = np.load(ifd_path)
        assert len(ifd_diffs) == len(conversations), \
            f"IFD count mismatch: {len(ifd_diffs)} != {len(conversations)}. Delete {ifd_path} to re-score."
    else:
        t0 = time.time()
        num_gpus = torch.cuda.device_count()
        print(f"IFD scoring with {num_gpus} GPU(s), model_tag={args.model_tag}")

        # Build per-GPU shards + count turn-pairs in one pass
        shards = [[] for _ in range(num_gpus)]
        shard_turn_counts = [0] * num_gpus
        for i, conv in enumerate(conversations):
            gpu = i % num_gpus
            shards[gpu].append((i, conv))
            shard_turn_counts[gpu] += _count_turns(conv["messages"])
        total_turns = sum(shard_turn_counts)
        total_batches = sum((c + args.ifd_batch_size - 1) // args.ifd_batch_size for c in shard_turn_counts)
        num_tok_workers = max(1, NUM_CPU_PROC // num_gpus)
        print(f"Turn-pairs: {total_turns:,}, batches: {total_batches:,} (batch_size={args.ifd_batch_size}, tok_workers/gpu={num_tok_workers})")

        result_dict = mp.Manager().dict() if num_gpus > 1 else {}
        per_gpu_args = [
            (gpu_id, shards[gpu_id], args.model_tag, args.ifd_batch_size,
             num_tok_workers, result_dict)
            for gpu_id in range(num_gpus)
        ]
        run_gpu_workers(ifd_worker, per_gpu_args, total_batches, "IFD scoring")

        # Merge results
        ifd_diffs = np.zeros(len(conversations), dtype=np.float32)
        for gpu_id in range(num_gpus):
            for conv_idx, score in result_dict[gpu_id].items():
                ifd_diffs[conv_idx] = score

        np.save(ifd_path, ifd_diffs)
        print(f"Saved IFD diffs to {ifd_path}")
        print(f"Stage 4 done in {time.time() - t0:.1f}s")

    # ----------
    _stage("Stage 5: Select")
    t0 = time.time()

    # Hard filter: remove IFD >= 0 (instruction hurts or doesn't help)
    valid_mask = ifd_diffs < 0
    n_valid = valid_mask.sum()
    print(f"Hard filter: removed {len(conversations) - n_valid:,} samples with IFD diff >= 0 ({len(conversations):,} -> {n_valid:,})")

    # Build per-cluster sorted lists of (conv_idx, ifd_diff), sorted ascending (most negative first).
    # More negative IFD diff = instruction helps more = stronger instruction-response coupling = better SFT data.
    cluster_pools = defaultdict(list)  # cluster_id -> list of (conv_idx, ifd_diff)
    for i in range(len(conversations)):
        if valid_mask[i]:
            cluster_pools[cluster_labels[i]].append((i, ifd_diffs[i]))

    # Sort each cluster: most negative IFD first (strongest instruction-response coupling)
    for c in cluster_pools:
        cluster_pools[c].sort(key=lambda x: x[1])

    num_active_clusters = len(cluster_pools)
    assert num_active_clusters > 0, "No samples passed IFD filter (all IFD diffs >= 0)"

    # Uniform allocation with iterative redistribution
    # Each cluster gets target_rows / num_clusters. If a cluster is too small,
    # its excess slots are redistributed to remaining clusters iteratively.
    allocations = {}  # cluster_id -> num slots
    base_alloc = args.target_rows // num_active_clusters
    remainder = args.target_rows % num_active_clusters
    for rank, c in enumerate(sorted(cluster_pools.keys())):
        allocations[c] = base_alloc + (1 if rank < remainder else 0)

    # Iteratively redistribute: if a cluster has fewer samples than its allocation,
    # the excess is spread evenly across clusters that still have room.
    while True:
        excess = 0
        oversized = []  # clusters with room to spare
        for c in sorted(allocations.keys()):
            pool_size = len(cluster_pools[c])
            if pool_size < allocations[c]:
                excess += allocations[c] - pool_size
                allocations[c] = pool_size  # cap to what's available
            elif pool_size > allocations[c]:
                oversized.append(c)
        if excess == 0 or not oversized:
            break
        # Distribute excess to oversized clusters
        extra_per = excess // len(oversized)
        extra_rem = excess % len(oversized)
        for j, c in enumerate(oversized):
            allocations[c] += extra_per + (1 if j < extra_rem else 0)

    # Greedy selection per cluster: iterate by IFD (best first), skip near-duplicates
    if args.min_embed_dist > 0:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_normed = embeddings / np.maximum(norms, 1e-8)
        sim_threshold = 1.0 - args.min_embed_dist
        print(f"Embedding dedup: cosine_sim > {sim_threshold:.2f} within each cluster")
    else:
        emb_normed = None

    n_deduped = 0
    selected_indices = []
    for c in sorted(allocations.keys()):
        pool = cluster_pools[c]
        take = allocations[c]
        cluster_embs = []  # embeddings of selected samples in this cluster
        cluster_selected = []
        for idx, _ in pool:
            if len(cluster_selected) >= take:
                break
            if emb_normed is not None and cluster_embs:
                sims = emb_normed[idx] @ np.array(cluster_embs).T
                if sims.max() > sim_threshold:
                    n_deduped += 1
                    continue
            cluster_selected.append(idx)
            if emb_normed is not None:
                cluster_embs.append(emb_normed[idx])
        selected_indices.extend(cluster_selected)

    if n_deduped > 0:
        print(f"Skipped {n_deduped:,} near-duplicates during selection")
    print(f"Selected {len(selected_indices):,} conversations (target: {args.target_rows:,})")
    print(f"Stage 5 done in {time.time() - t0:.1f}s")

    # Free embeddings
    del embeddings
    gc.collect()

    # ----------
    _stage("Stage 6: Save")
    t0 = time.time()

    output_path = os.path.join(curated_dir, f"{args.output_name}.jsonl")
    selected_indices.sort()
    with open(output_path, "w") as f:
        for idx in selected_indices:
            f.write(json.dumps(conversations[idx]) + "\n")
    print(f"Saved {len(selected_indices):,} conversations to {output_path}")

    # ----------
    _stage("Stats")

    selected_clusters = cluster_labels[selected_indices]
    unique, counts = np.unique(selected_clusters, return_counts=True)
    print(f"Clusters used: {len(unique)}/{args.num_clusters}")
    print(f"Cluster sizes — min: {counts.min()}, median: {int(np.median(counts))}, max: {counts.max()}")

    selected_ifd = ifd_diffs[selected_indices]
    p10, p50, p90 = np.percentile(selected_ifd, [10, 50, 90])
    print(f"IFD diff — p10: {p10:.4f}, p50: {p50:.4f}, p90: {p90:.4f}")
    print(f"IFD diff — mean: {selected_ifd.mean():.4f}, min: {selected_ifd.min():.4f}, max: {selected_ifd.max():.4f}")

    print(f"\nTotal selected: {len(selected_indices):,}")
    print(f"Output: {output_path}")
    print(f"Stage 6 done in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    # Line-buffer stdout so prints appear immediately before slow operations
    sys.stdout.reconfigure(line_buffering=True)
    main()
