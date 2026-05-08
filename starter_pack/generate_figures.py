"""
generate_figures.py — Generează grafice pentru lucrare din trainer_state.json și fișiere de rezultate.

Output: 6 figuri PNG salvate în results/figures/

Usage:
    python generate_figures.py \
        --checkpoints_dir /path/to/checkpoints \
        --results_dir /path/to/results \
        --output_dir /path/to/results/figures
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configurare globală: stiluri și culori consistente
# ---------------------------------------------------------------------------

MARKERS = ['Action', 'Actor', 'Effect', 'Evidence', 'Victim']

EXP_CONFIGS = [
    ('exp01a_full', 'Full FT (66M)', '#2E86AB'),       # albastru
    ('exp01b_unfreeze4', 'Unfreeze 4 (28M)', '#E63946'),  # roșu
    ('exp01c_linear', 'Linear Probe (1.5K)', '#06A77D'),  # verde
]

# Stil consistent
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})


# ---------------------------------------------------------------------------
# Helper-i pentru încărcare date
# ---------------------------------------------------------------------------

def find_latest_checkpoint(marker_dir):
    """Găsește cel mai recent checkpoint în folderul markerului."""
    if not os.path.exists(marker_dir):
        return None
    contents = os.listdir(marker_dir)
    ckpts = [c for c in contents if c.startswith('checkpoint-')]
    if not ckpts:
        return None
    return sorted(ckpts, key=lambda x: int(x.split('-')[1]))[-1]


def load_trainer_state(checkpoints_dir, exp_id, marker):
    """Încarcă trainer_state.json din cel mai recent checkpoint."""
    marker_dir = os.path.join(checkpoints_dir, exp_id, marker)
    latest = find_latest_checkpoint(marker_dir)
    if latest is None:
        return None
    ts_path = os.path.join(marker_dir, latest, 'trainer_state.json')
    if not os.path.exists(ts_path):
        return None
    with open(ts_path) as f:
        return json.load(f)


def extract_train_eval_curves(trainer_state):
    """
    Din trainer_state.json extrage:
    - train_steps, train_losses (la fiecare ~50 steps)
    - eval_epochs, eval_losses, eval_f1s (la fiecare epocă)
    """
    train_steps, train_losses = [], []
    eval_epochs, eval_losses, eval_f1s = [], [], []
    
    for entry in trainer_state.get('log_history', []):
        if 'loss' in entry and 'eval_loss' not in entry:
            # Train log entry
            train_steps.append(entry.get('step', 0))
            train_losses.append(entry['loss'])
        elif 'eval_loss' in entry:
            # Eval log entry
            eval_epochs.append(entry.get('epoch', 0))
            eval_losses.append(entry['eval_loss'])
            eval_f1s.append(entry.get('eval_f1', 0))
    
    return train_steps, train_losses, eval_epochs, eval_losses, eval_f1s


def load_scores_json(checkpoints_dir, exp_id, scores_filename='scores_val_full.json'):
    """Încarcă scores JSON de la eval_token.py."""
    path = os.path.join(checkpoints_dir, exp_id, scores_filename)
    if not os.path.exists(path):
        # Fallback la scores_val.json (pentru multi-seed)
        path = os.path.join(checkpoints_dir, exp_id, 'scores_val.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_qualitative_analysis(results_dir, exp_id):
    """Încarcă analiza calitativă JSON."""
    path = os.path.join(results_dir, 'qualitative', f'analysis_{exp_id}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figura 1: Curbe de loss (training + eval) per categorie pentru 1A
# ---------------------------------------------------------------------------

def figure_1_loss_curves(checkpoints_dir, output_dir):
    """Loss training + eval pentru exp01a_full, 5 categorii."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=False)
    
    for idx, marker in enumerate(MARKERS):
        ax = axes[idx]
        ts = load_trainer_state(checkpoints_dir, 'exp01a_full', marker)
        if ts is None:
            ax.set_title(f'{marker}\n(no data)')
            continue
        
        train_steps, train_losses, eval_epochs, eval_losses, _ = extract_train_eval_curves(ts)
        
        # Convertim train_steps la "epoch unit" pentru a fi pe aceeași scară
        steps_per_epoch = max(train_steps) / max(eval_epochs) if eval_epochs else 1
        train_epochs_continuous = [s / steps_per_epoch for s in train_steps]
        
        ax.plot(train_epochs_continuous, train_losses, color='#2E86AB',
                linewidth=1, alpha=0.6, label='Train loss')
        ax.plot(eval_epochs, eval_losses, color='#E63946',
                linewidth=2, marker='o', markersize=5, label='Val loss')
        
        ax.set_title(marker, fontweight='bold')
        ax.set_xlabel('Epoch')
        if idx == 0:
            ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.suptitle('Figura 1. Curbe de loss (training și validation) — Full Fine-Tuning',
                 fontweight='bold', y=1.05)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fig1_loss_curves.png')
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path}")


# ---------------------------------------------------------------------------
# Figura 2: F1 eval pe epoci, comparând 3 strategii
# ---------------------------------------------------------------------------

def figure_2_f1_progression(checkpoints_dir, output_dir):
    """F1 eval pe epoci pentru 1A vs 1B vs 1C, 5 categorii."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharey=True)
    
    for idx, marker in enumerate(MARKERS):
        ax = axes[idx]
        
        for exp_id, label, color in EXP_CONFIGS:
            ts = load_trainer_state(checkpoints_dir, exp_id, marker)
            if ts is None:
                continue
            _, _, eval_epochs, _, eval_f1s = extract_train_eval_curves(ts)
            ax.plot(eval_epochs, eval_f1s, color=color, linewidth=2,
                    marker='o', markersize=5, label=label)
        
        ax.set_title(marker, fontweight='bold')
        ax.set_xlabel('Epoch')
        if idx == 0:
            ax.set_ylabel('F1 (eval, intern)')
            ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 0.8)
    
    plt.suptitle('Figura 2. Evoluția F1 pe validation pe parcursul antrenării',
                 fontweight='bold', y=1.05)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fig2_f1_progression.png')
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path}")


# ---------------------------------------------------------------------------
# Figura 3: Bar chart F1 final per categorie × strategie
# ---------------------------------------------------------------------------

def figure_3_f1_comparison(checkpoints_dir, output_dir):
    """Bar chart F1 token-based final pentru 3 strategii × 5 categorii."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    n_markers = len(MARKERS)
    x = np.arange(n_markers)
    width = 0.27
    
    for i, (exp_id, label, color) in enumerate(EXP_CONFIGS):
        scores = load_scores_json(checkpoints_dir, exp_id)
        if scores is None:
            continue
        f1s = [scores[f'F1_{m}_Token'] for m in MARKERS]
        
        bars = ax.bar(x + (i - 1) * width, f1s, width, label=label, color=color, alpha=0.85)
        # Adaugăm valoarea pe bar
        for b, v in zip(bars, f1s):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f'{v:.3f}', ha='center', fontsize=8)
    
    ax.set_xlabel('Categorie')
    ax.set_ylabel('F1 token-based (IoU ≥ 0.5)')
    ax.set_title('Figura 3. F1 final pe validation — comparație strategii fine-tuning',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MARKERS)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.55)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fig3_f1_comparison.png')
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path}")


# ---------------------------------------------------------------------------
# Figura 4: Multi-seed F1 cu error bars
# ---------------------------------------------------------------------------

def figure_4_multiseed(checkpoints_dir, output_dir):
    """Bar chart F1 mediu ± std pe 3 seeds, doar pentru 1A."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    seed_exps = [
        ('exp01a_full', 'scores_val_full.json'),
        ('exp01a_full_seed123', 'scores_val.json'),
        ('exp01a_full_seed2024', 'scores_val.json'),
    ]
    
    # Strângem F1 pentru fiecare marker, pentru fiecare seed
    f1_per_marker = {m: [] for m in MARKERS}
    for exp_id, scores_file in seed_exps:
        scores = load_scores_json(checkpoints_dir, exp_id, scores_file)
        if scores is None:
            continue
        for m in MARKERS:
            f1_per_marker[m].append(scores[f'F1_{m}_Token'])
    
    means = [np.mean(f1_per_marker[m]) for m in MARKERS]
    stds = [np.std(f1_per_marker[m], ddof=1) for m in MARKERS]
    
    x = np.arange(len(MARKERS))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color='#2E86AB', alpha=0.85,
                  error_kw={'elinewidth': 2, 'ecolor': '#1A1A2E'})
    
    # Valori pe bar
    for b, mean, std in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + std + 0.01,
                f'{mean:.3f} ± {std:.3f}', ha='center', fontsize=9)
    
    ax.set_xlabel('Categorie')
    ax.set_ylabel('F1 token-based (IoU ≥ 0.5)')
    ax.set_title('Figura 4. Reproductibilitate Full FT — F1 mediu ± std peste 3 seeds (42, 123, 2024)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MARKERS)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.55)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fig4_multiseed.png')
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path}")


# ---------------------------------------------------------------------------
# Figura 5: Distribuția erorilor stacked bar
# ---------------------------------------------------------------------------

def figure_5_error_distribution(results_dir, output_dir):
    """Stacked bar cu TP/FP/FN_partial/FN_missed per categorie pentru 1A."""
    analysis = load_qualitative_analysis(results_dir, 'exp01a_full')
    if analysis is None:
        print("  SKIP fig5: nu există analysis_exp01a_full.json")
        return
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    x = np.arange(len(MARKERS))
    width = 0.6
    
    tp = np.array([analysis['summary'][m]['TP'] for m in MARKERS])
    fp = np.array([analysis['summary'][m]['FP'] for m in MARKERS])
    fnp = np.array([analysis['summary'][m]['FN_partial'] for m in MARKERS])
    fnm = np.array([analysis['summary'][m]['FN_missed'] for m in MARKERS])
    
    ax.bar(x, tp, width, label='TP (corect)', color='#06A77D')
    ax.bar(x, fp, width, bottom=tp, label='FP (inventat)', color='#E63946')
    ax.bar(x, fnp, width, bottom=tp+fp, label='FN_partial (aproape)', color='#FFB627')
    ax.bar(x, fnm, width, bottom=tp+fp+fnp, label='FN_missed (ratat)', color='#7B7B7B')
    
    # Total deasupra
    totals = tp + fp + fnp + fnm
    for i, total in enumerate(totals):
        ax.text(i, total + 30, f'n={total}', ha='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Categorie')
    ax.set_ylabel('Număr cazuri')
    ax.set_title('Figura 5. Distribuția erorilor pe validation — Full FT',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MARKERS)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fig5_error_distribution.png')
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path}")


# ---------------------------------------------------------------------------
# Figura 6: Lungime span gold vs predicted
# ---------------------------------------------------------------------------

def figure_6_span_lengths(results_dir, output_dir):
    """Bar chart paired cu lungimea medie a span-urilor gold vs pred per categorie."""
    fig, ax = plt.subplots(figsize=(11, 5))
    
    analyses = {}
    for exp_id, _, _ in EXP_CONFIGS:
        a = load_qualitative_analysis(results_dir, exp_id)
        if a is not None:
            analyses[exp_id] = a
    
    x = np.arange(len(MARKERS))
    width = 0.18
    
    # Gold (din 1A — același peste experimente)
    gold_lens = [analyses['exp01a_full']['summary'][m]['avg_gold_span_len_chars']
                 for m in MARKERS]
    ax.bar(x - 1.5*width, gold_lens, width, label='Gold', color='#1A1A2E', alpha=0.9)
    
    # Predicții pentru fiecare strategie
    for i, (exp_id, label, color) in enumerate(EXP_CONFIGS):
        if exp_id not in analyses:
            continue
        pred_lens = [analyses[exp_id]['summary'][m]['avg_pred_span_len_chars']
                     for m in MARKERS]
        ax.bar(x + (i - 0.5) * width, pred_lens, width, label=f'Pred {label}',
               color=color, alpha=0.85)
    
    ax.set_xlabel('Categorie')
    ax.set_ylabel('Lungime medie span (caractere)')
    ax.set_title('Figura 6. Sub-extragere sistematică — lungime span gold vs predicții',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MARKERS)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fig6_span_lengths.png')
    plt.savefig(out_path)
    plt.close()
    print(f"  ✓ {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_dir", required=True,
                   help="Folder cu cele 3+2 experimente (exp01a_full etc.)")
    p.add_argument("--results_dir", required=True,
                   help="Folder cu results/ (qualitative/ și CSV-uri)")
    p.add_argument("--output_dir", required=True,
                   help="Folder unde se salvează figurile PNG")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generez figurile pentru lucrare")
    print("="*60 + "\n")
    
    print("[1/6] Loss curves...")
    figure_1_loss_curves(args.checkpoints_dir, args.output_dir)
    
    print("[2/6] F1 progression...")
    figure_2_f1_progression(args.checkpoints_dir, args.output_dir)
    
    print("[3/6] F1 comparison bar chart...")
    figure_3_f1_comparison(args.checkpoints_dir, args.output_dir)
    
    print("[4/6] Multi-seed error bars...")
    figure_4_multiseed(args.checkpoints_dir, args.output_dir)
    
    print("[5/6] Error distribution stacked bar...")
    figure_5_error_distribution(args.results_dir, args.output_dir)
    
    print("[6/6] Span lengths comparison...")
    figure_6_span_lengths(args.results_dir, args.output_dir)
    
    print(f"\n✓ Toate figurile au fost salvate în: {args.output_dir}")


if __name__ == "__main__":
    main()
