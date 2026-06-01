# Licență SemEval 2026 Task 10 (PsyCoMark) — Subtask Span Extraction

**Autor:** Miriam Costea

## Descrierea proiectului

Acest repo conține implementarea și experimentele pentru lucrarea de licență
pe subtask-ul **span extraction** al competiției SemEval 2026 Task 10 (PsyCoMark) —
identificarea de pasaje (spans) din texte care exprimă markeri psiho-comportamentali
și conspirativi (Action, Actor, Effect, Evidence, Victim).

## Structura repo-ului
.
├── starter_pack/              Cod modificat din pachetul oficial SemEval
│   ├── train_one_span.py      Antrenare per categorie (one-vs-rest)
│   ├── infer_one_span.py      Inferență, generează submission JSONL
│   ├── eval_token.py          Evaluator oficial (token F1 cu IoU≥0.5)
│   ├── train_one_span_lora.py   Antrenare LoRA pe Qwen2.5 (Etapa 2)
│   ├── infer_one_span_lora.py   Inferență cu adaptoare LoRA (Etapa 2)
│   ├── qualitative_analysis.py  Analiza erorilor (TP/FP/FN_partial/FN_missed)
│   └── generate_figures.py    Grafice pentru lucrare (6 figuri PNG)
│
├── data/                      Split-uri reproductibile (vezi data/README.md)
│   └── rehydrated/
│       └── val_split*.jsonl
│
├── results/                   Rezultate experimentale și figuri
│   ├── etapa1_all_runs.csv          Tabel principal F1 (3 strategii × 5 markeri)
│   ├── etapa1_multiseed_1A.csv      Multi-seed cu mean ± std
│   ├── qualitative/                 Analiza calitativă (JSON + exemple)
│   ├── figures/                     6 figuri PNG pentru lucrare
│   ├── scores/                      Scoruri evaluate Etapa 1 (Codabench format)
│   └── stage2/                      Scoruri Etapa 2 (multi-seed + ablație rang)
│       ├── scores_val_seed{42,123,2024}.json
│       ├── summary_stage2.json      Sumar consolidat (medii, std, comparații)
│       └── ablation_rank/           Ablație rang LoRA pe Victim
│
├── etapa1_documentare.md      Documentare completă Etapa 1 (cu interpretări)
├── README.md                  Acest fișier
└── .gitignore
## Etapele lucrării

### Etapa 1 — DistilBERT cu strategii fine-tuning (FINALIZAT)

Investigare a 3 strategii pe 5 categorii:
- **Linear Probe** (1.5K params) — F1 macro 0.055
- **Unfreeze Last 4** (28M params) — F1 macro 0.253
- **Full Fine-Tuning** (66M params) — F1 macro 0.297 ± 0.006 (multi-seed pe 3 seeds)

Detalii complete în `etapa1_documentare.md`.

### Etapa 2 — LoRA pe Qwen2.5-0.5B (FINALIZAT)

Adaptare eficientă parametric (LoRA) a modelului Qwen2.5-0.5B pe aceeași sarcină,
cu doar 0.87% parametri antrenabili (4.3M / 498M).

- **Configurație:** r=32, alpha=64, target q/k/v/o_proj, 12 epoci, class_weight=3.0
- **F1 macro:** 0.2997 ± 0.0079 (multi-seed pe 3 seeds)
- **Ablație rang LoRA** (Victim): r=4 -> 0.2845, r=32 -> 0.3302, r=64 -> 0.3345

Echivalență statistică cu Full Fine-Tuning DistilBERT (Etapa 1: 0.2970), dar cu
o reducere de ~15x a parametrilor actualizați. Detalii în `etapa2_documentare.md`.

## Cum rulezi experimentele

```bash
# Antrenare per strategie
python starter_pack/train_one_span.py \
    --data_path data/rehydrated/train_rehydrated.jsonl \
    --output_dir checkpoints/exp01a_full \
    --marker_types Action Actor Effect Evidence Victim \
    --unfreeze_last_n 6 \
    --num_epochs 10 \
    --seed 42

# Inferență pe validation
python starter_pack/infer_one_span.py \
    --model_dir checkpoints/exp01a_full \
    --test_file data/rehydrated/val_split.jsonl \
    --submission_file checkpoints/exp01a_full/val_submission.jsonl \
    --marker_types Action Actor Effect Evidence Victim

# Evaluare oficială
python starter_pack/eval_token.py \
    --ground_truth_file data/rehydrated/val_split.jsonl \
    --prediction_file checkpoints/exp01a_full/val_submission.jsonl \
    --scores_output_file checkpoints/exp01a_full/scores_val.json

# Analiza calitativă
python starter_pack/qualitative_analysis.py \
    --gold_file data/rehydrated/val_split.jsonl \
    --pred_dir checkpoints \
    --output_dir results/qualitative \
    --experiments exp01a_full exp01b_unfreeze4 exp01c_linear

# Generare grafice pentru lucrare
python starter_pack/generate_figures.py \
    --checkpoints_dir checkpoints \
    --results_dir results \
    --output_dir results/figures
```

## Reproducerea rezultatelor

Toate experimentele folosesc seed fix (42 by default, cu multi-seed pe 123 și 2024
pentru verificare statistică). Pentru reproducere completă, vezi `data/README.md`
pentru obținerea datasetului oficial PsyCoMark.

