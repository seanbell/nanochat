#!/usr/bin/env python3
"""
Analyze fullsweep results by extracting metrics directly from log files.
Works even when CSV rows haven't been written yet (Phase 1 in-progress).

Usage:
    python runs/analyze_fullsweep.py [SERIES_NAME]        # print analysis
    python runs/analyze_fullsweep.py baseline2 --write-csv # also write/update CSVs
"""
import argparse
import csv
import os
import re
import glob
import sys

NANOCHAT_BASE_DIR = os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))

# Chat eval tasks and their random baselines (for ChatCORE)
CHAT_TASKS = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
CHAT_BASELINES = {'ARC-Easy': 25.0, 'ARC-Challenge': 25.0, 'MMLU': 25.0, 'GSM8K': 0.0, 'HumanEval': 0.0, 'SpellingBee': 0.0}


# =============================================================================
# Log file parsing
# =============================================================================

def read_log(path):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return ""

def find_all(pattern, text):
    """Return all regex matches as a list of group(1) strings."""
    return [m.group(1) for m in re.finditer(pattern, text)]

def find_last_float(pattern, text):
    """Return the last float match, or None."""
    matches = find_all(pattern, text)
    return float(matches[-1]) if matches else None


def parse_pretrain_log(text):
    """Extract metrics from a pretrain log."""
    if not text:
        return {}
    m = {}
    # Parameter counts
    total = re.search(r'(?m)^total\s*:\s*([\d,]+)', text)
    if total:
        m['num_params'] = int(total.group(1).replace(',', ''))
    scaling = re.search(r'(?m)^transformer_matrices\s*:\s*([\d,]+)', text)
    if scaling:
        m['num_scaling_params'] = int(scaling.group(1).replace(',', ''))
    # Training setup
    tokens = re.search(r'Total number of training tokens: ([\d,]+)', text)
    if tokens:
        m['tokens_trained'] = int(tokens.group(1).replace(',', ''))
    flops = re.search(r'Total training FLOPs estimate: ([\d.]+e[+\d]+)', text)
    if flops:
        m['total_flops'] = flops.group(1)
    # Final metrics (take last occurrence)
    m['pt_train_loss'] = find_last_float(r'step \d+/\d+.*?\| loss: ([\d.]+)', text)
    m['pt_val_bpb'] = find_last_float(r'Validation bpb: ([\d.]+)', text)
    m['pt_core'] = find_last_float(r'CORE metric: ([\d.]+)', text)
    m['pt_core_bpb'] = find_last_float(r'CORE BPB: ([\d.]+)', text)
    # Wall time
    m['pt_time_min'] = find_last_float(r'total time: ([\d.]+)m', text)
    # Progress
    steps = re.findall(r'step (\d+)/(\d+)', text)
    if steps:
        m['pt_current_step'] = int(steps[-1][0])
        m['pt_total_steps'] = int(steps[-1][1])
    return m


def parse_base_eval_log(text):
    """Extract metrics from a base_eval log."""
    if not text:
        return {}
    m = {}
    m['be_train_bpb'] = find_last_float(r'train bpb: ([\d.]+)', text)
    m['be_val_bpb'] = find_last_float(r'val bpb: ([\d.]+)', text)
    m['be_core'] = find_last_float(r'CORE metric: ([\d.]+)', text)
    m['be_core_bpb'] = find_last_float(r'CORE BPB: ([\d.]+)', text)
    # Per-task CORE results
    tasks = []
    for match in re.finditer(r'Evaluating: (\S+) .*?accuracy: ([\d.]+) \| centered: ([-\d.]+)(?: \| bpb: ([\d.]+))?', text):
        tasks.append({
            'task': match.group(1),
            'accuracy': float(match.group(2)),
            'centered': float(match.group(3)),
            'bpb': float(match.group(4)) if match.group(4) else None,
        })
    if tasks:
        m['core_tasks'] = tasks
    return m


def parse_sft_log(text):
    if not text:
        return {}
    m = {}
    m['sft_train_loss'] = find_last_float(r'\| loss: ([\d.]+)', text)
    m['sft_val_bpb'] = find_last_float(r'Validation bpb: ([\d.]+)', text)
    m['sft_time_min'] = find_last_float(r'total time: ([\d.]+)m', text)
    return m


def parse_chat_eval_log(text):
    """Extract per-task chat eval results (task -> accuracy %)."""
    if not text:
        return {}
    tasks = {}
    for match in re.finditer(r'(\S+) accuracy: ([\d.]+)%', text):
        tasks[match.group(1)] = float(match.group(2))
    return tasks


def parse_chat_eval_bpb_log(text):
    """Extract per-task BPB from chat eval log (task -> bpb)."""
    if not text:
        return {}
    bpbs = {}
    for match in re.finditer(r'(\S+) accuracy: [\d.]+% \| bpb: ([\d.]+)', text):
        bpbs[match.group(1)] = float(match.group(2))
    return bpbs


def parse_holdout_bpb_log(text):
    if not text:
        return {}
    return {
        'train_bpb': find_last_float(r'train bpb: ([\d.]+)', text),
        'val_bpb': find_last_float(r'val bpb: ([\d.]+)', text),
    }


def parse_rl_log(text):
    if not text:
        return {}
    m = {}
    rewards = re.findall(r'Average reward: ([\d.]+)', text)
    if rewards:
        m['rl_final_reward'] = float(rewards[-1])
    steps = re.findall(r'Step (\d+)/(\d+)', text)
    if steps:
        m['rl_current_step'] = int(steps[-1][0])
        m['rl_total_steps'] = int(steps[-1][1])
    return m


# =============================================================================
# Main extraction
# =============================================================================

def find_series(name=None):
    pattern = os.path.join(NANOCHAT_BASE_DIR, "*_fullsweep_results")
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not dirs:
        print("No fullsweep results found.")
        sys.exit(1)
    if name:
        match = [d for d in dirs if name in os.path.basename(d)]
        if not match:
            print(f"No results matching '{name}'. Available: {[os.path.basename(d) for d in dirs]}")
            sys.exit(1)
        return match[-1]
    return dirs[-1]


def get_depth_metrics(results_dir, series_name, depth):
    """Extract all available metrics for a given depth from log files."""
    tag = f"{series_name}_fullsweep_d{depth}"
    log = lambda name: read_log(os.path.join(results_dir, f"{tag}_{name}.log"))

    m = {}
    m.update(parse_pretrain_log(log('pretrain')))
    m.update(parse_base_eval_log(log('base_eval')))
    m.update(parse_sft_log(log('sft')))
    chat_eval_sft_text = log('chat_eval_sft')
    m['chat_sft'] = parse_chat_eval_log(chat_eval_sft_text)
    m['chat_sft_bpb'] = parse_chat_eval_bpb_log(chat_eval_sft_text)
    sft_holdout_text = log('sft_holdout_bpb')
    sft_forget = parse_holdout_bpb_log(sft_holdout_text)
    sft_forget_eval = parse_base_eval_log(sft_holdout_text)
    if sft_forget:
        m['sft_forget_val_bpb'] = sft_forget.get('val_bpb')
    if sft_forget_eval:
        m['sft_forget_core'] = sft_forget_eval.get('be_core')
        m['sft_forget_core_bpb'] = sft_forget_eval.get('be_core_bpb')
        if 'core_tasks' in sft_forget_eval:
            m['sft_forget_core_tasks'] = sft_forget_eval['core_tasks']
    m.update(parse_rl_log(log('rl')))
    chat_eval_rl_text = log('chat_eval_rl')
    m['chat_rl'] = parse_chat_eval_log(chat_eval_rl_text)
    m['chat_rl_bpb'] = parse_chat_eval_bpb_log(chat_eval_rl_text)
    rl_holdout_text = log('rl_holdout_bpb')
    rl_forget = parse_holdout_bpb_log(rl_holdout_text)
    rl_forget_eval = parse_base_eval_log(rl_holdout_text)
    if rl_forget:
        m['rl_forget_val_bpb'] = rl_forget.get('val_bpb')
    if rl_forget_eval:
        m['rl_forget_core'] = rl_forget_eval.get('be_core')
        m['rl_forget_core_bpb'] = rl_forget_eval.get('be_core_bpb')
        if 'core_tasks' in rl_forget_eval:
            m['rl_forget_core_tasks'] = rl_forget_eval['core_tasks']

    return m


def compute_chatcore(accs):
    """ChatCORE from task -> accuracy% dict. Returns None if any task is missing."""
    if any(accs.get(t) is None for t in CHAT_BASELINES):
        return None
    centered = [(accs[t] - CHAT_BASELINES[t]) / (100.0 - CHAT_BASELINES[t]) for t in CHAT_BASELINES]
    return sum(centered) / len(centered)


# =============================================================================
# CSV writing
# =============================================================================

def write_csvs(results_dir, depths, all_metrics):
    """Write results.csv and results_core_detail.csv from extracted metrics."""
    results_file = os.path.join(results_dir, "results.csv")
    core_detail_file = os.path.join(results_dir, "results_core_detail.csv")

    # --- Main results CSV (one row per depth) ---
    # Build rows as dicts — column set is dynamic
    rows = []
    for d in depths:
        m = all_metrics[d]
        # Skip depths that haven't finished pretraining
        val_bpb = m.get('be_val_bpb') or m.get('pt_val_bpb')
        if val_bpb is None:
            continue

        core_score = m.get('be_core') or m.get('pt_core')
        core_bpb = m.get('be_core_bpb') or m.get('pt_core_bpb')
        pt_time_sec = int(m['pt_time_min'] * 60) if m.get('pt_time_min') else None
        sft_time_sec = int(m['sft_time_min'] * 60) if m.get('sft_time_min') else None
        total_time = sum(t for t in [pt_time_sec, sft_time_sec] if t is not None) or None

        # Param:data ratio
        tokens = m.get('tokens_trained')
        scaling = m.get('num_scaling_params')
        param_data_ratio = round(tokens / scaling, 2) if tokens and scaling else None

        row = {
            'depth': d,
            'model_dim': d * 64,
            'num_params': m.get('num_params'),
            'num_scaling_params': scaling,
            'pretrain_iters': m.get('pt_total_steps'),
            'tokens_trained': tokens,
            'total_flops': m.get('total_flops'),
            'param_data_ratio': param_data_ratio,
            'val_bpb': val_bpb,
            'core_score': core_score,
            'core_bpb': core_bpb,
            'pretrain_time_sec': pt_time_sec,
            'sft_val_bpb': m.get('sft_val_bpb'),
            'sft_forget_bpb': m.get('sft_forget_val_bpb'),
            'sft_forget_core': m.get('sft_forget_core'),
            'sft_forget_core_bpb': m.get('sft_forget_core_bpb'),
            'sft_time_sec': sft_time_sec,
        }

        # Chat eval tasks (SFT and RL) — dynamic from CHAT_TASKS
        chat_sft = m.get('chat_sft', {})
        chat_sft_bpb = m.get('chat_sft_bpb', {})
        for task in CHAT_TASKS:
            row[f'{task.lower().replace("-", "_")}_sft'] = chat_sft.get(task)
            row[f'{task.lower().replace("-", "_")}_sft_bpb'] = chat_sft_bpb.get(task)
        row['chatcore_sft'] = compute_chatcore(chat_sft)

        row['rl_final_reward'] = m.get('rl_final_reward')
        row['rl_forget_bpb'] = m.get('rl_forget_val_bpb')
        row['rl_forget_core'] = m.get('rl_forget_core')
        row['rl_forget_core_bpb'] = m.get('rl_forget_core_bpb')

        chat_rl = m.get('chat_rl', {})
        chat_rl_bpb = m.get('chat_rl_bpb', {})
        for task in CHAT_TASKS:
            row[f'{task.lower().replace("-", "_")}_rl'] = chat_rl.get(task)
            row[f'{task.lower().replace("-", "_")}_rl_bpb'] = chat_rl_bpb.get(task)
        row['chatcore_rl'] = compute_chatcore(chat_rl)

        row['total_time_sec'] = total_time
        rows.append(row)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {results_file}")

    # --- Per-task CORE detail CSV ---
    detail_rows = []
    for d in depths:
        m = all_metrics[d]
        for task in m.get('core_tasks', []):
            detail_rows.append({
                'depth': d,
                'task': task['task'],
                'accuracy': task['accuracy'],
                'centered': task['centered'],
                'bpb': task['bpb'] if task['bpb'] is not None else 'NaN',
            })
        core = m.get('be_core') or m.get('pt_core')
        core_bpb = m.get('be_core_bpb') or m.get('pt_core_bpb')
        if core is not None:
            detail_rows.append({
                'depth': d, 'task': 'CORE', 'accuracy': '',
                'centered': core, 'bpb': core_bpb if core_bpb is not None else 'NaN',
            })

    if detail_rows:
        with open(core_detail_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['depth', 'task', 'accuracy', 'centered', 'bpb'])
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f"Wrote {len(detail_rows)} rows to {core_detail_file}")


# =============================================================================
# Display — color support
# =============================================================================

# ANSI colors chosen for Solarized Light readability
_COLOR = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if _COLOR else str(text)
def green(t):    return _c(t, '32')
def red(t):      return _c(t, '31')
def cyan(t):     return _c(t, '36')
def bold(t):     return _c(t, '1')
def dim(t):      return _c(t, '2')
def bold_red(t): return _c(t, '1;31')

def _visible_len(s):
    """String length excluding ANSI escape codes."""
    return len(re.sub(r'\033\[[0-9;]*m', '', str(s)))

def _pad(text, width, align='<'):
    padding = max(0, width - _visible_len(text))
    return (' ' * padding + str(text)) if align == '>' else (str(text) + ' ' * padding)

# =============================================================================
# Display — formatting helpers
# =============================================================================

def section(title):
    print(f"\n{bold('=' * 80)}\n{bold(title)}\n{bold('=' * 80)}")

def fmt(val, decimals=4):
    if val is None: return dim("—")
    if isinstance(val, float): return f"{val:.{decimals}f}"
    return str(val)

def fmt_pct(val):
    return f"{val:.2f}%" if val is not None else dim("—")

def delta_str(new, old, dec=4, color=True):
    if new is None or old is None: return dim('—')
    d = new - old
    s = f"{'+' if d >= 0 else ''}{d:.{dec}f}"
    return s  # caller handles color

def print_table(headers, rows):
    all_rows = [headers] + rows
    widths = [max(_visible_len(str(cell)) for cell in col) for col in zip(*all_rows)]
    aligns = ['<'] + ['>'] * (len(headers) - 1)
    # Header row in bold
    print(" | ".join(_pad(bold(str(h)), w, a) for h, w, a in zip(headers, widths, aligns)))
    print(dim("-+-".join("-" * w for w in widths)))
    for row in rows:
        print(" | ".join(_pad(str(c), w, a) for c, w, a in zip(row, widths, aligns)))


def scaling_table(depths, all_metrics, getters, dec=4, better='lower'):
    """Print metric-x-depth table with colored deltas. Regressions in red."""
    headers = ['metric', f'd{depths[0]}'] + [f'd{d} (delta)' for d in depths[1:]]
    rows = []
    for name, getter in getters:
        vals = [getter(all_metrics[d]) for d in depths]
        row = [name, fmt(vals[0], dec)]
        for i in range(1, len(vals)):
            if vals[i] is not None and vals[i-1] is not None:
                d = vals[i] - vals[i-1]
                good = (d < 0 and better == 'lower') or (d > 0 and better == 'higher')
                delta_text = f"({'+' if d >= 0 else ''}{d:.{dec}f})"
                if good:
                    row.append(f"{fmt(vals[i], dec)} {green(delta_text)}")
                else:
                    # Regression in scaling = surprise, make it stand out
                    row.append(f"{fmt(vals[i], dec)} {bold_red(delta_text)}")
            else:
                row.append(fmt(vals[i], dec))
        rows.append(row)
    print_table(headers, rows)


def display_results(results_dir, series_name, depths, all_metrics):
    print(f"Series: {bold(series_name)}")
    print(f"Dir: {dim(results_dir)}")
    if not depths:
        print("No depth logs found.")
        return

    def is_done(d, stage):
        tag = f"{series_name}_fullsweep_d{d}"
        return os.path.exists(os.path.join(results_dir, f"{tag}_{stage}.done"))
    def has_log_file(d, stage):
        tag = f"{series_name}_fullsweep_d{d}"
        return os.path.exists(os.path.join(results_dir, f"{tag}_{stage}.log"))

    STAGES = ['pretrain', 'base_eval', 'sft', 'chat_eval_sft', 'sft_holdout_bpb',
              'rl', 'chat_eval_rl', 'rl_holdout_bpb']

    # --- Stage completion ---
    section("STAGE COMPLETION")
    rows = []
    for d in depths:
        row = [str(d)]
        for s in STAGES:
            if is_done(d, s):
                row.append(green("done"))
            elif has_log_file(d, s):
                row.append(cyan("run"))
            else:
                row.append(dim("—"))
        rows.append(row)
    print_table(['depth'] + STAGES, rows)

    # --- Model overview ---
    section("MODEL OVERVIEW")
    rows = []
    for d in depths:
        m = all_metrics[d]
        progress = ""
        if m.get('pt_current_step') and m.get('pt_total_steps') and not is_done(d, 'pretrain'):
            pct = 100 * m['pt_current_step'] / m['pt_total_steps']
            cur, tot = m['pt_current_step'], m['pt_total_steps']
            progress = " " + cyan(f"({cur}/{tot} {pct:.0f}%)")
        params = m.get('num_params')
        rows.append([
            str(d), str(d * 64),
            f"{params:,}" if params else dim("—"),
            f"{m['tokens_trained']:,}" if m.get('tokens_trained') else dim("—"),
            m.get('total_flops', dim('—')),
            (f"{m['pt_time_min']:.1f}m" if m.get('pt_time_min') else dim("—")) + progress,
        ])
    print_table(['depth', 'dim', 'params', 'tokens', 'FLOPs', 'pt_time'], rows)

    eval_depths = [d for d in depths if all_metrics[d].get('be_val_bpb') is not None]

    # Helper: get per-task CORE data as {task_name: {field: value}}
    def core_task_getter(task_name, field):
        def getter(m):
            for t in m.get('core_tasks', []):
                if t['task'] == task_name:
                    return t.get(field)
            return None
        return getter

    # Discover CORE task names from first depth that has them
    core_task_names = []
    for d in eval_depths:
        if all_metrics[d].get('core_tasks'):
            core_task_names = [t['task'] for t in all_metrics[d]['core_tasks']]
            break

    # =========================================================================
    # SCALING: BPB across depths
    # =========================================================================
    if eval_depths:
        section("SCALING: BPB (across depths, lower is better)")
        bpb_getters = [
            ('PT train loss', lambda m: m.get('pt_train_loss')),
            ('PT train BPB',  lambda m: m.get('be_train_bpb')),
            ('PT val BPB',    lambda m: m.get('be_val_bpb') or m.get('pt_val_bpb')),
            ('CORE BPB',      lambda m: m.get('be_core_bpb') or m.get('pt_core_bpb')),
            ('SFT train loss',lambda m: m.get('sft_train_loss')),
            ('SFT val BPB',   lambda m: m.get('sft_val_bpb')),
            ('SFT→PT val',    lambda m: m.get('sft_forget_val_bpb')),
            ('SFT→PT CORE BPB', lambda m: m.get('sft_forget_core_bpb')),
            ('RL→PT val',     lambda m: m.get('rl_forget_val_bpb')),
            ('RL→PT CORE BPB', lambda m: m.get('rl_forget_core_bpb')),
        ]
        # Per-task CORE BPB
        for tname in core_task_names:
            bpb_getters.append((tname, core_task_getter(tname, 'bpb')))
        # Chat eval per-task BPB (SFT, RL)
        for source, key in [('sft', 'chat_sft_bpb'), ('rl', 'chat_rl_bpb')]:
            sample = next((all_metrics[d].get(key, {}) for d in eval_depths if all_metrics[d].get(key)), {})
            for task in sample:
                bpb_getters.append((f'{task} ({source})',
                    lambda m, k=key, t=task: (m.get(k) or {}).get(t)))
        scaling_table(eval_depths, all_metrics, bpb_getters, dec=4, better='lower')

    # =========================================================================
    # SCALING: Accuracy across depths
    # =========================================================================
    if eval_depths:
        section("SCALING: ACCURACY (across depths, higher is better)")
        acc_getters = [
            ('CORE (base eval)', lambda m: m.get('be_core') or m.get('pt_core')),
            ('CORE (sft→pt)', lambda m: m.get('sft_forget_core')),
            ('CORE (rl→pt)', lambda m: m.get('rl_forget_core')),
        ]
        # Per-task CORE accuracy
        for tname in core_task_names:
            acc_getters.append((tname, core_task_getter(tname, 'accuracy')))
        # Chat eval tasks (SFT, RL)
        for source, key in [('sft', 'chat_sft'), ('rl', 'chat_rl')]:
            sample = next((all_metrics[d].get(key, {}) for d in eval_depths if all_metrics[d].get(key)), {})
            for task in sample:
                acc_getters.append((f'{task} ({source})',
                    lambda m, k=key, t=task: (m.get(k) or {}).get(t)))
            if sample:
                acc_getters.append((f'ChatCORE ({source})',
                    lambda m, k=key: compute_chatcore(m.get(k, {}))))
        scaling_table(eval_depths, all_metrics, acc_getters, dec=4, better='higher')

    # =========================================================================
    # PIPELINE: BPB progression (PT → SFT → RL)
    # =========================================================================
    pipeline_depths = [d for d in depths if all_metrics[d].get('sft_forget_val_bpb') is not None]
    if pipeline_depths:
        section("PIPELINE: BPB (how pretraining knowledge degrades through fine-tuning)")
        has_rl = any(all_metrics[d].get('rl_forget_val_bpb') for d in pipeline_depths)
        headers = ['depth', 'PT val', 'SFT→PT val', 'forget']
        if has_rl:
            headers += ['RL→PT val', 'total forget']
        rows = []
        for d in pipeline_depths:
            m = all_metrics[d]
            pt = m.get('be_val_bpb') or m.get('pt_val_bpb')
            sft = m.get('sft_forget_val_bpb')
            forget_sft = red(delta_str(sft, pt)) if sft and pt else dim('—')
            row = [str(d), fmt(pt), fmt(sft), forget_sft]
            if has_rl:
                rl = m.get('rl_forget_val_bpb')
                forget_rl = red(delta_str(rl, pt)) if rl and pt else dim('—')
                row += [fmt(rl), forget_rl]
            rows.append(row)
        print_table(headers, rows)

    # =========================================================================
    # PIPELINE: CORE accuracy (PT → SFT → RL)
    # =========================================================================
    pipeline_core_depths = [d for d in depths if all_metrics[d].get('sft_forget_core') is not None]
    if pipeline_core_depths:
        section("PIPELINE: CORE ACCURACY (how pretraining accuracy changes through fine-tuning)")
        has_rl = any(all_metrics[d].get('rl_forget_core') for d in pipeline_core_depths)
        headers = ['depth', 'PT CORE', 'SFT→PT CORE', 'delta']
        if has_rl:
            headers += ['RL→PT CORE', 'delta']
        rows = []
        for d in pipeline_core_depths:
            m = all_metrics[d]
            pt = m.get('be_core') or m.get('pt_core')
            sft = m.get('sft_forget_core')
            delta_sft = delta_str(sft, pt)
            if sft is not None and pt is not None:
                delta_sft = green(delta_sft) if sft >= pt else red(delta_sft)
            row = [str(d), fmt(pt), fmt(sft), delta_sft]
            if has_rl:
                rl = m.get('rl_forget_core')
                delta_rl = delta_str(rl, pt)
                if rl is not None and pt is not None:
                    delta_rl = green(delta_rl) if rl >= pt else red(delta_rl)
                row += [fmt(rl), delta_rl]
            rows.append(row)
        print_table(headers, rows)

    # =========================================================================
    # PIPELINE: Task accuracy progression (base → SFT → RL)
    # =========================================================================
    # Map base eval task names to chat eval names (arc_easy ↔ ARC-Easy, etc.)
    task_map = {}  # base_name -> chat_name
    for d in depths:
        base_names = {t['task'] for t in all_metrics[d].get('core_tasks', [])}
        chat_names = set(all_metrics[d].get('chat_sft', {}).keys())
        for bn in base_names:
            for cn in chat_names:
                if cn.lower().replace('-', '_') == bn.lower():
                    task_map[bn] = cn

    pipeline_chat_depths = [d for d in depths if all_metrics[d].get('chat_sft')]
    if pipeline_chat_depths:
        section("PIPELINE: TASK ACCURACY")
        print(bold_red("WARNING") + ": base and sft/rl use different eval formats — deltas are NOT apples-to-apples!")
        print(f"  base = CORE eval: {cyan('10-shot')}, completion format (rank full answer text by perplexity)")
        print(f"  sft/rl = chat eval: {cyan('0-shot')}, chat format (constrained to letter token logits)")
        has_rl = any(all_metrics[d].get('chat_rl') for d in pipeline_chat_depths)

        # Discover all chat tasks (preserving order from first depth)
        all_chat_tasks = []
        for d in pipeline_chat_depths:
            for t in all_metrics[d].get('chat_sft', {}):
                if t not in all_chat_tasks:
                    all_chat_tasks.append(t)

        headers = ['depth', 'task', 'base(10s,compl)', 'sft(0s,chat)', 'delta']
        if has_rl:
            headers += ['rl(0s,chat)', 'delta(rl-sft)']

        rows = []
        for d in pipeline_chat_depths:
            m = all_metrics[d]
            core = {t['task']: t for t in m.get('core_tasks', [])}
            chat_sft = m.get('chat_sft', {})
            chat_rl = m.get('chat_rl', {})

            for chat_name in all_chat_tasks:
                # Match base eval task (completion format, accuracy in [0,1])
                base_name = next((bn for bn, cn in task_map.items() if cn == chat_name), None)
                base_acc = core.get(base_name, {}).get('accuracy') if base_name else None
                base_pct = base_acc * 100 if base_acc is not None else None
                sft_acc = chat_sft.get(chat_name)

                # Color the delta: these are cross-format so dim them
                d_val = delta_str(sft_acc, base_pct, 2)
                d_colored = dim(d_val) if base_pct is not None and sft_acc is not None else d_val

                row = [str(d), chat_name, fmt_pct(base_pct), fmt_pct(sft_acc), d_colored]
                if has_rl:
                    rl_acc = chat_rl.get(chat_name)
                    # SFT→RL is apples-to-apples, color normally
                    rl_d = delta_str(rl_acc, sft_acc, 2)
                    if rl_acc is not None and sft_acc is not None:
                        rl_d = green(rl_d) if rl_acc >= sft_acc else red(rl_d)
                    row += [fmt_pct(rl_acc), rl_d]
                rows.append(row)

        print_table(headers, rows)

    # =========================================================================
    # RL training progress
    # =========================================================================
    rl_depths = [d for d in depths if all_metrics[d].get('rl_final_reward') is not None
                 or all_metrics[d].get('rl_current_step') is not None]
    if rl_depths:
        section("RL TRAINING")
        rows = []
        for d in rl_depths:
            m = all_metrics[d]
            progress = cyan(f"{m['rl_current_step']}/{m['rl_total_steps']}") if m.get('rl_current_step') else green('done')
            rows.append([str(d), progress, fmt(m.get('rl_final_reward')), fmt(m.get('rl_forget_val_bpb'))])
        print_table(['depth', 'progress', 'final_reward', 'forget_bpb'], rows)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze fullsweep results")
    parser.add_argument('series', nargs='?', default=None, help="Series name to analyze")
    parser.add_argument('--write-csv', action='store_true', help="Write/update results CSVs from log files")
    parser.add_argument('--color', action='store_true', help="Force colored output (even when piping)")
    args = parser.parse_args()

    global _COLOR
    if args.color:
        _COLOR = True

    results_dir = find_series(args.series)
    series_name = os.path.basename(results_dir).replace("_fullsweep_results", "")

    # Discover depths from log files
    depths = sorted(set(
        int(m.group(1))
        for f in os.listdir(results_dir)
        if (m := re.search(r'_d(\d+)_', f))
    ))

    # Extract all metrics
    all_metrics = {d: get_depth_metrics(results_dir, series_name, d) for d in depths}

    # Display
    display_results(results_dir, series_name, depths, all_metrics)

    # Write CSVs
    if args.write_csv:
        write_csvs(results_dir, depths, all_metrics)


if __name__ == "__main__":
    main()
