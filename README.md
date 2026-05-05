# Licență SemEval 2026 Task 10 — Span Extraction

Lucrare de licență pe subtask-ul de span extraction din SemEval 2026 Task 10:
PsyCoMark — Psycholinguistic Conspiracy Marker Extraction and Detection.

## Setup
- Pornit de la starter pack oficial: https://github.com/hide-ous/semeval26_task10_starter_pack
- Antrenare pe Google Colab T4 (15GB VRAM)

## Etape experimentale
1. **Baseline DistilBERT** — reproducerea rezultatelor oficiale
2. **DistilBERT cu unfreeze parțial** — antrenare ultimelor N layere transformer
3. **LLM modern + LoRA** — Qwen2.5-0.5B cu adaptori LoRA

## Structura
- `starter_pack/` — codul de antrenare/inferență (oficial + modificările mele)
- `notebooks/` — notebook-uri Colab pentru experimente
- `results/` — metrici per experiment
