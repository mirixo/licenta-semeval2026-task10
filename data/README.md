# Date pentru SemEval 2026 Task 10 (PsyCoMark) — Subtask Span Extraction

Acest folder documentează modul de obținere a datelor. Textul comentariilor nu este redistribuit în acest depozit, conform termenilor datasetului oficial PsyCoMark și ai platformei Reddit.

## Fișiere

Datasetul oficial este distribuit sub formă de metadate (id-uri Reddit, subreddit, adnotator), fără text. Textul se recuperează local prin rehidratare. După rehidratare rezultă:

- `rehydrated/train_rehydrated.jsonl` — set de antrenare (4316 exemple, cu text complet)

Acest fișier **nu este inclus** în depozit, fiind supus licenței oficiale.

## Cum obții datele

1. Acceptă termenii datasetului PsyCoMark pe platforma oficială a competiției (Codabench).
2. Descarcă fișierul cu metadate `train_redacted.jsonl` (conține doar id-uri Reddit, subreddit și adnotator — fără text).
3. Rehidratează textul cu scriptul oficial, care necesită credențiale Reddit API:
   ```bash
   python ../starter_pack/rehydrate_data.py
   ```
   Rezultatul, `rehydrated/train_rehydrated.jsonl`, conține textul complet și este folosit de scripturile de antrenare.

## Partiționarea train / validare

Partiționarea 80/20 (antrenare / validare) se realizează automat în scripturile de antrenare, fără *seed* fixat. Partiția de validare efectiv folosită de fiecare rulare este salvată în folderul de output al rulării, ca `val_split_used.jsonl`, astfel încât evaluarea se face pe exact setul de exemple ținut deoparte la antrenare.
