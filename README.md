# SemEval 2026 Task 10 (PsyCoMark) — Extragerea markerilor psiho-comportamentali

**Autor:** Miriam Costea
**Lucrare de licență** — subtask-ul de *span extraction*

Acest depozit conține codul și experimentele lucrării de licență pe subtask-ul de extragere a *span*-urilor din competiția SemEval 2026 Task 10 (PsyCoMark): identificarea, în comentarii de pe Reddit, a pasajelor care exprimă cinci categorii de markeri psiho-comportamentali din discursul conspirativ — Action, Actor, Effect, Evidence, Victim.

Sunt investigate două direcții metodologice complementare: ajustarea fină (*fine-tuning*) a modelului encoder DistilBERT și adaptarea eficientă parametric (LoRA) a modelului decoder Qwen2.5-0.5B, pe aceeași sarcină și cu aceleași metrici.

## Rezultate principale

Metrica oficială este F1 la nivel de token, cu un prag de suprapunere IoU ≥ 0,5. Fiecare configurație principală a fost rulată de cinci ori, fără *seed* fixat; se raportează media și deviația standard.

|Configurație|Parametri antrenabili|F1 macro|
|-|-|-|
|DistilBERT — *linear probing*|0,15 mil.|0,055|
|DistilBERT — *unfreeze* 4 straturi|21,30 mil.|0,253|
|DistilBERT — *fine-tuning* complet|66,90 mil.|**0,2916 ± 0,0079**|
|Qwen2.5-0.5B — doar capul (fără LoRA)|\~1,8 mii|0,1197 ± 0,0048|
|Qwen2.5-0.5B — LoRA (r = 32)|4,30 mil. (0,87%)|**0,2930 ± 0,0149**|
|Baseline oficial (pachet de start)|—|\~0,15|

Cele două abordări principale sunt echivalente statistic (diferență de 0,0014, sub deviația standard a fiecăreia), deși LoRA actualizează de \~15 ori mai puțini parametri. Ambele aproape dublează baseline-ul oficial.

## Structura depozitului

```
.
├── README.md                  acest fișier
├── data/
│   └── README.md              cum se obțin datele (textul nu este redistribuit)
├── starter\_pack/              cod (modificat din pachetul oficial SemEval)
│   ├── train\_one\_span.py          antrenare DistilBERT (one-vs-rest)
│   ├── infer\_one\_span.py          inferență DistilBERT
│   ├── train\_one\_span\_lora.py     antrenare Qwen2.5 + LoRA
│   ├── infer\_one\_span\_lora.py     inferență Qwen2.5 (+ LoRA sau doar cap)
│   ├── eval\_token.py              evaluator oficial (F1 token, IoU ≥ 0,5)
│   ├── qualitative\_analysis.py    analiza erorilor (TP / FP / FN)
│   ├── generate\_figures.py        generarea figurilor
│   ├── rehydrate\_data.py          rehidratare date (script oficial)
│   └── requirements.txt           dependențe
└── results/                   scoruri, sinteze și figuri
```

## Metodologie de evaluare

Toate experimentele folosesc schema one-vs-rest: câte un model binar independent per categorie (cinci capuri de clasificare separate), aleasă pentru că adnotările conțin *span*-uri suprapuse, pe care un singur clasificator multi-class nu le poate modela.

Variabilitatea rezultatelor este estimată prin **rulări repetate fără *seed* fixat:** fiecare rulare pornește de la o inițializare și o partiționare aleatoare proprie, iar partiția de validare efectiv folosită este salvată pe disc (`val\_split\_used.jsonl`), astfel încât evaluarea se face întotdeauna pe exact setul ținut deoparte la antrenare. Se raportează media și deviația standard pe cinci rulări.

## Date

Textul comentariilor nu este redistribuit în acest depozit, conform termenilor datasetului oficial PsyCoMark și ai platformei Reddit. Se distribuie doar metadate (id-uri Reddit), iar textul se recuperează local prin rehidratare. Pașii compleți sunt în [`data/README.md`](data/README.md). Pe scurt:

1. Acceptarea termenilor datasetului pe platforma oficială a competiției.
2. Descărcarea fișierelor cu metadate (`train\_redacted.jsonl`).
3. Rehidratarea cu scriptul oficial, care necesită credențiale Reddit API:

```bash
   python starter\_pack/rehydrate\_data.py
   ```

   Rezultatul este `data/rehydrated/train\_rehydrated.jsonl`.

   ## Instalare

   ```bash
pip install -r starter\_pack/requirements.txt
```

   ## Rulare

   ### Etapa 1 — DistilBERT (*fine-tuning* complet)

   ```bash
# Antrenare (one-vs-rest, cele cinci categorii)
python starter\_pack/train\_one\_span.py \\
    --data\_path data/rehydrated/train\_rehydrated.jsonl \\
    --output\_dir runs/distilbert\_full \\
    --marker\_types Action Actor Effect Evidence Victim \\
    --unfreeze\_last\_n 6 --num\_epochs 10

# Inferență pe partiția de validare salvată de antrenare
python starter\_pack/infer\_one\_span.py \\
    --model\_dir runs/distilbert\_full \\
    --test\_file runs/distilbert\_full/val\_split\_used.jsonl \\
    --submission\_file runs/distilbert\_full/val\_submission.jsonl \\
    --marker\_types Action Actor Effect Evidence Victim

# Evaluare oficială (F1 token, IoU ≥ 0,5)
python starter\_pack/eval\_token.py \\
    --ground\_truth\_file runs/distilbert\_full/val\_split\_used.jsonl \\
    --prediction\_file runs/distilbert\_full/val\_submission.jsonl \\
    --scores\_output\_file runs/distilbert\_full/scores\_val.json
```

   Strategiile de *fine-tuning* se obțin variind `--unfreeze\_last\_n`: `0` = *linear probing* (doar capul), `4` = *unfreeze* parțial, `6` = *fine-tuning* complet.

   ### Etapa 2 — Qwen2.5-0.5B + LoRA

   ```bash
python starter\_pack/train\_one\_span\_lora.py \\
    --data\_path data/rehydrated/train\_rehydrated.jsonl \\
    --output\_dir runs/qwen\_lora \\
    --marker\_types Action Actor Effect Evidence Victim \\
    --lora\_r 32 --lora\_alpha 64 --num\_epochs 12 --class\_weight\_positive 3.0

python starter\_pack/infer\_one\_span\_lora.py \\
    --model\_dir runs/qwen\_lora \\
    --test\_file runs/qwen\_lora/val\_split\_used.jsonl \\
    --submission\_file runs/qwen\_lora/val\_submission.jsonl \\
    --marker\_types Action Actor Effect Evidence Victim

python starter\_pack/eval\_token.py \\
    --ground\_truth\_file runs/qwen\_lora/val\_split\_used.jsonl \\
    --prediction\_file runs/qwen\_lora/val\_submission.jsonl \\
    --scores\_output\_file runs/qwen\_lora/scores\_val.json
```

   ### Estimarea variabilității (rulări repetate)

   Pentru media și deviația standard raportate, comanda de antrenare se repetă de cinci ori, fără *seed*. Fiecare rulare salvează propria partiție de validare; scorurile per rulare se agregă la final.

   ## Ablații

   ```bash
# Fără ponderarea clasei pozitive (verifică utilitatea ponderii 3,0)
python starter\_pack/train\_one\_span\_lora.py ... --class\_weight\_positive 1.0

# Fără LoRA: model de bază înghețat, se antrenează doar capul de clasificare
python starter\_pack/train\_one\_span\_lora.py ... --no\_lora

# Variația rangului LoRA
python starter\_pack/train\_one\_span\_lora.py ... --lora\_r 4
python starter\_pack/train\_one\_span\_lora.py ... --lora\_r 64
```

   ## Licență și atribuire

   Codul din `starter\_pack/` este adaptat din pachetul oficial de start al competiției SemEval 2026 Task 10. Datasetul PsyCoMark aparține organizatorilor competiției și este supus termenilor proprii de utilizare; textul comentariilor nu este redistribuit aici.

