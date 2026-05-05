# Starter Pack — Cod modificat pentru lucrarea de licență

Acest folder conține codul de antrenare și inferență pentru subtask-ul de
**span extraction** din SemEval 2026 Task 10 (PsyCoMark).

## Origine

Fișierele oficiale provin din:
https://github.com/hide-ous/semeval26_task10_starter_pack

## Modificări față de versiunea oficială

### `train_one_span.py`

1. **argparse complet** — toți hiperparametrii sunt configurabili din linie de comandă
2. **Split 80/20 train/val reproductibil** cu `--seed=42`
3. **Tokeni speciali primesc label `-100`** (ignorați de loss) — fix pentru bug-ul
   în care `[CLS]`, `[SEP]`, `[PAD]` erau antrenați ca "non-marker"
4. **`compute_metrics`** cu F1 (clasa pozitivă), precision, recall, F1 micro, F1 macro
5. **Linear LR scheduler cu warmup 10%**
6. **Early stopping** cu `patience` configurabil (default 3)
7. **`load_best_model_at_end=True`** + `metric_for_best_model="f1"` + `greater_is_better=True`
8. **`save_total_limit=2`** + `evaluation_strategy="epoch"` + `save_strategy="epoch"`
9. **Funcția `configure_trainable_layers(model, unfreeze_last_n)`** —
   dezgheață doar ultimele N layere transformer din DistilBERT
10. **Salvare explicită** a modelului final în `<output_dir>/<marker_type>/final_model/`
11. **Salvare metrici** ca JSON în `<output_dir>/metrics.json` pentru tabelul comparativ

### `infer_one_span.py`

1. **argparse complet**
2. **`resolve_model_path()`** — caută în ordine: `final_model/` → `checkpoint-XXX` (cel mai mare) → folder bază
3. **`max_length` configurabil** (consistent cu antrenarea)
4. **`AutoModel/AutoTokenizer`** în loc de `DistilBert*` specific (pregătit pentru Etapa 2 cu LLM)

## Utilizare

### Antrenare baseline reprodus (full fine-tuning, ca în varianta oficială)

```bash
python train_one_span.py \
    --data_path /content/drive/MyDrive/Licenta_SemEval2026_Task10/data/rehydrated/train_rehydrated.jsonl \
    --output_dir /content/drive/MyDrive/Licenta_SemEval2026_Task10/checkpoints/exp01a_full \
    --marker_types Victim Evidence \
    --unfreeze_last_n 6 \
    --num_epochs 10 \
    --seed 42
```

### Antrenare cu unfreeze parțial (4 din 6 layere)

```bash
python train_one_span.py \
    --data_path /content/drive/MyDrive/Licenta_SemEval2026_Task10/data/rehydrated/train_rehydrated.jsonl \
    --output_dir /content/drive/MyDrive/Licenta_SemEval2026_Task10/checkpoints/exp01b_unfreeze4 \
    --marker_types Victim Evidence \
    --unfreeze_last_n 4 \
    --num_epochs 10 \
    --seed 42
```

### Antrenare linear probing (doar capul, restul înghețat)

```bash
python train_one_span.py \
    --data_path /content/drive/MyDrive/Licenta_SemEval2026_Task10/data/rehydrated/train_rehydrated.jsonl \
    --output_dir /content/drive/MyDrive/Licenta_SemEval2026_Task10/checkpoints/exp01c_linear \
    --marker_types Victim Evidence \
    --unfreeze_last_n 0 \
    --num_epochs 15 \
    --learning_rate 1e-3 \
    --seed 42
```

(Linear probing necesită LR mai mare și mai multe epoci — capul are
mult mai puțini parametri, deci convergență mai înceată.)

### Inferență

```bash
python infer_one_span.py \
    --model_dir /content/drive/MyDrive/Licenta_SemEval2026_Task10/checkpoints/exp01a_full \
    --test_file /content/drive/MyDrive/Licenta_SemEval2026_Task10/data/rehydrated/dev_rehydrated.jsonl \
    --submission_file /content/drive/MyDrive/Licenta_SemEval2026_Task10/results/exp01a_submission.jsonl \
    --marker_types Victim Evidence
```

### Evaluare (cu scriptul oficial)

```bash
python eval_token.py \
    --gold /content/drive/MyDrive/Licenta_SemEval2026_Task10/data/rehydrated/dev_rehydrated.jsonl \
    --pred /content/drive/MyDrive/Licenta_SemEval2026_Task10/results/exp01a_submission.jsonl
```
