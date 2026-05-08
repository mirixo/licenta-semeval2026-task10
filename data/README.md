# Date pentru SemEval 2026 Task 10 (PsyCoMark)

## Structura folder-ului
data/
├── README.md (acest fișier)
└── rehydrated/
├── val_split.jsonl              (863 sample-uri, seed=42)
├── val_split_seed123.jsonl      (863 sample-uri, seed=123, multi-seed)
└── val_split_seed2024.jsonl     (863 sample-uri, seed=2024, multi-seed)
## Fișiere disponibile în acest repo

Doar **split-urile de validare** sunt incluse, deoarece ele sunt parte integrantă
din setup-ul experimental (split reproductibil cu seed fix).

## Fișiere NU incluse (sub licență oficială)

Următoarele fișiere sunt parte din datasetul oficial SemEval 2026 Task 10
și **nu sunt redistribuite** în acest repo:

- `train_rehydrated.jsonl` — set de antrenare (4316 sample-uri)
- `dev_rehydrated.jsonl` — set de testare oficial (100 sample-uri, gold ascuns)

## Cum obții datele oficiale

1. Înregistrare pe Codabench:
   https://www.codabench.org/competitions/[task-id]/
2. Acceptare termeni și condiții ai dataset-ului PsyCoMark
3. Descărcare `train_redacted.jsonl` și `dev_redacted.jsonl` (acestea conțin doar
   metadate — id-uri Reddit, subreddit, anotator — fără text)
4. Rehidratare prin scriptul oficial `rehydrate_data.py` (necesită credentiale
   Reddit API pentru a recupera textul de la postările originale)

## Cum regenerezi val_splits cu un seed nou

După ce ai `train_rehydrated.jsonl` în acest folder, poți regenera split-urile:

```python
import json, random

def split_train_val(data, val_ratio, seed):
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_val = int(len(data) * val_ratio)
    val_idx = set(indices[:n_val])
    train = [data[i] for i in range(len(data)) if i not in val_idx]
    val = [data[i] for i in range(len(data)) if i in val_idx]
    return train, val

with open('train_rehydrated.jsonl') as f:
    data = [json.loads(line) for line in f]

_, val = split_train_val(data, val_ratio=0.2, seed=42)
with open('val_split.jsonl', 'w') as f:
    for ex in val:
        f.write(json.dumps(ex) + '\n')
```

## Note tehnice

- Toate split-urile au 863 sample-uri (20% din 4316 train).
- Seed-ul controlează atât random shuffle al indices, cât și inițializarea modelului
  (vezi `train_one_span.py --seed`).
- Diferite seed-uri produc split-uri diferite — important pentru analiza
  reproductibilității (cf. multi-seed în Etapa 1, raportat în
  `results/etapa1_multiseed_1A.csv`).
