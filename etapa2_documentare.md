# Etapa II — Adaptare LoRA pe Qwen2.5-0.5B

## Obiectiv

Investigarea adaptării eficiente parametric (LoRA) a unui model lingvistic
recent (Qwen2.5-0.5B) pe sarcina de extragere a markerilor psiho-comportamentali,
ca alternativă la fine-tuning-ul complet al unui model BERT distilat (Etapa I).

## Configurație finală

Configurație uniformă pentru toți cei cinci markeri:

| Hiperparametru | Valoare |
|---|---|
| Model de bază | Qwen/Qwen2.5-0.5B |
| Rang LoRA (r) | 32 |
| Alpha | 64 |
| Dropout | 0.1 |
| Module țintă | q_proj, k_proj, v_proj, o_proj |
| Epoci | 12 |
| Rată de învățare | 1e-4 |
| Dimensiune batch | 8 |
| Lungime maximă secvență | 128 |
| Pondere clasă pozitivă | 3.0 |
| Parametri antrenabili | 4,328,964 / 498,361,732 (0.87%) |

## Rezultate multi-seed (F1 token, IoU >= 0.5)

Evaluare pe partiție internă de validare (863 comentarii), trei seed-uri (42, 123, 2024).

| Marker | seed=42 | seed=123 | seed=2024 | Medie | Std |
|---|---|---|---|---|---|
| Action | 0.2673 | 0.2727 | 0.2738 | 0.2713 | 0.0035 |
| Actor | 0.4239 | 0.4019 | 0.4137 | 0.4132 | 0.0110 |
| Effect | 0.2152 | 0.2381 | 0.2465 | 0.2333 | 0.0162 |
| Evidence | 0.2165 | 0.2532 | 0.2465 | 0.2387 | 0.0195 |
| Victim | 0.3302 | 0.3492 | 0.3465 | 0.3420 | 0.0103 |
| **F1 Macro** | 0.2906 | 0.3030 | 0.3054 | **0.2997** | **0.0079** |
| F1 Micro | 0.3050 | 0.3135 | 0.3125 | 0.3103 | 0.0046 |

## Ablație asupra rangului LoRA (marker Victim, seed=42)

| Rang (r) | Alpha | Parametri antrenabili | F1 Victim |
|---|---|---|---|
| 4 | 8 | 544,260 (0.11%) | 0.2845 |
| 32 | 64 | 4,328,964 (0.87%) | 0.3302 |
| 64 | 128 | 8,654,340 (1.72%) | 0.3345 |

Saltul mare 4 -> 32 (+0.046) și platoul 32 -> 64 (+0.004) justifică alegerea r=32
ca punct de echilibru între capacitate și eficiență parametrică.

## Comparație Etapa I vs Etapa II (medii multi-seed)

| Marker | Etapa I (DistilBERT FT) | Etapa II (Qwen + LoRA) | Delta |
|---|---|---|---|
| Action | 0.2320 | 0.2713 | +0.039 |
| Actor | 0.4360 | 0.4132 | -0.023 |
| Effect | 0.2370 | 0.2333 | -0.004 |
| Evidence | 0.2230 | 0.2387 | +0.016 |
| Victim | 0.3300 | 0.3420 | +0.012 |
| **F1 Macro** | 0.2970 | 0.2997 | +0.003 |
| F1 Micro | 0.3050 | 0.3103 | +0.005 |

Concluzie: echivalență statistică (diferență sub deviația standard) cu o reducere
de ~15x a parametrilor actualizați, demonstrând eficiența parametrică a LoRA.

## Reproductibilitate

Antrenare per marker:

```bash
python train_one_span_lora.py \
    --data_path data/rehydrated/train_rehydrated.jsonl \
    --output_dir checkpoints/exp02_qwen_lora \
    --marker_types Victim \
    --target_modules q_proj k_proj v_proj o_proj \
    --lora_r 32 --lora_alpha 64 --num_epochs 12 \
    --learning_rate 1e-4 --class_weight_positive 3.0 --seed 42
```

Inferență și evaluare:

```bash
python infer_one_span_lora.py \
    --model_dir checkpoints/exp02_qwen_lora \
    --test_file data/rehydrated/val_split.jsonl \
    --submission_file checkpoints/exp02_qwen_lora/val_submission.jsonl \
    --marker_types Action Actor Effect Evidence Victim \
    --base_model Qwen/Qwen2.5-0.5B

python eval_token.py \
    --ground_truth_file data/rehydrated/val_split.jsonl \
    --prediction_file checkpoints/exp02_qwen_lora/val_submission.jsonl \
    --scores_output_file checkpoints/exp02_qwen_lora/scores_val.json
```

## Note

- Modelele antrenate (adaptoarele LoRA) nu sunt incluse în repository din cauza dimensiunii.
- Datele oficiale PsyCoMark nu sunt redistribuite (licență SemEval).
- Scorurile complete în format JSON sunt disponibile în results/stage2/.
