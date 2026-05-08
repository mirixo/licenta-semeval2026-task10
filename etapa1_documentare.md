# Etapa 1: DistilBERT pe SemEval 2026 Task 10 (PsyCoMark) — Subtask Span Extraction

**Documentare experimentală pentru lucrare de licență**
**Autor:** Miriam Costea
**Coordonator:** [Numele profesorului]
**Data:** Mai 2026

---

## 1. Obiectiv și context

SemEval 2026 Task 10 (PsyCoMark) propune analiza markerilor psiho-comportamentali și conspirativi din texte engleze. Subtask-ul abordat — **span extraction** — cere identificarea pasajelor (span-urilor) din text care exprimă unul din cei 5 markeri:

- **Action** — acțiune sau comportament (ex: "upset", "permitting ontological hatred")
- **Actor** — agent al unei acțiuni (ex: "Germany", "Redditors", "US")
- **Effect** — consecințe sau rezultate (ex: "events surrounding repatriation")
- **Evidence** — sursa probelor sau argumentelor (ex: "These surveys", "great article")
- **Victim** — subiect al acțiunii (ex: "Bolivia", "unvaccinated", "Gabbard")

Scopul Etapei 1 a fost reproducerea baseline-ului oficial (DistilBERT-base-uncased) și investigarea **strategiilor de fine-tuning** prin variația numărului de layere transformer dezghețate, conform indicațiilor coordonatorului.

## 2. Setup experimental

### 2.1 Date

- **Dataset oficial:** PsyCoMark train (4316 sample-uri rehidratate cu text complet)
- **Split:** 80% train (3453) / 20% validation (863), seed=42 reproducibil
- **Test/dev oficial:** 100 sample-uri cu gold ascuns (folosit doar pentru submisia finală pe Codabench)

### 2.2 Model și arhitectură

- **Model:** DistilBERT-base-uncased (66M parametri)
  - 6 layere transformer (vs 12 la BERT-base)
  - Tokenizare WordPiece, vocabular 30522
  - Cap de classification adăugat pe DistilBertForTokenClassification
- **Strategie de etichetare:** schemă binară (one-vs-rest), un model independent per categorie
  - Etichete per token: 0 = O (out), 1 = I (in marker)
  - Token-uri speciale ([CLS], [SEP], [PAD]) primesc -100 (ignorate de loss)

### 2.3 Hiperparametri

| Parametru | Valoare |
|---|---|
| Batch size | 16 |
| Max sequence length | 128 |
| Learning rate | 2e-5 (full/unfreeze), 1e-3 (linear probe) |
| Epochs | 10 (15 pentru linear probe) |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| LR scheduler | Linear cu warmup |
| Early stopping patience | 3 epochs |
| Best metric | F1 micro pe validation |

### 2.4 Strategii de fine-tuning evaluate

| Strategie | Cod | Layere antrenabile | Parametri antrenabili |
|---|---|---|---|
| **Full Fine-Tuning** | exp01a_full | toate 6 layere transformer + cap | 66,364,418 (100%) |
| **Unfreeze Last 4** | exp01b_unfreeze4 | ultimele 4 layere transformer + cap | 28,353,026 (43%) |
| **Linear Probe** | exp01c_linear | doar capul de classification | 1,538 (0.002%) |

### 2.5 Evaluare

- **Metrică oficială:** F1 token-based cu IoU ≥ 0.5 (Codabench evaluator)
- **Tokenizare pentru evaluare:** whitespace (consistent cu eval_token.py oficial)
- **Raportăm:** Precision, Recall, F1 per categorie + agregate macro și micro

## 3. Rezultate principale

### 3.1 Tabel consolidat — F1 token-based pe val_split (seed=42)

| Marker | Linear Probe | Unfreeze 4 | Full FT | Δ (Full vs Unf4) |
|---|---|---|---|---|
| **Action** | 0.0315 | 0.1996 | **0.2324** | +0.033 |
| **Actor** | 0.1952 | 0.3925 | **0.4362** | +0.044 |
| **Effect** | 0.0054 | 0.1994 | **0.2368** | +0.037 |
| **Evidence** | 0.0057 | 0.1849 | **0.2230** | +0.038 |
| **Victim** | 0.0363 | 0.2896 | **0.3298** | +0.040 |
| **F1 Macro** | 0.0548 | 0.2532 | **0.2916** | +0.039 |
| **F1 Micro** | 0.0841 | 0.2690 | **0.3046** | +0.036 |

**Observații cheie:**
- Saltul **Linear → Unfreeze 4** e dramatic (×4.6 pe macro), confirmând necesitatea fine-tuning-ului
- Saltul **Unfreeze 4 → Full FT** e modest dar consistent (~0.04 F1 pe toate categoriile)
- Constanța saltului ultim sugerează că **primele 2 layere transformer învață features generice utile uniform pentru toate tipurile de marker**

### 3.2 Multi-seed pe Full FT (verificare reproductibilitate)

Rezultate F1 pentru exp01a_full antrenate cu 3 seed-uri diferite (42, 123, 2024) — split-ul de date diferă pentru fiecare seed:

| Marker | Seed 42 | Seed 123 | Seed 2024 | **Mean ± Std** |
|---|---|---|---|---|
| **Action** | 0.2324 | 0.2722 | 0.2348 | **0.2465 ± 0.022** |
| **Actor** | 0.4362 | 0.4338 | 0.4339 | **0.4346 ± 0.001** |
| **Effect** | 0.2368 | 0.2401 | 0.2185 | **0.2318 ± 0.012** |
| **Evidence** | 0.2230 | 0.2424 | 0.2560 | **0.2405 ± 0.017** |
| **Victim** | 0.3298 | 0.3291 | 0.3360 | **0.3316 ± 0.004** |
| **F1 Macro** | 0.2916 | 0.3035 | 0.2958 | **0.2970 ± 0.006** |
| **F1 Micro** | 0.3046 | 0.3186 | 0.3086 | **0.3106 ± 0.007** |

**Observații cheie:**
- **Stabilitate ridicată în general** — deviația standard pe metricile agregate (0.006-0.007) e sub 1%, sugerând rezultate reproductibile
- **Diferența 1A vs 1B (0.297 vs 0.253) e statistic semnificativă** — diferența 0.044 e de 7-8× mai mare decât deviația standard 0.006
- **Variabilitate inegală pe categorii:**
  - Categorii "ușoare" (Actor: ±0.001, Victim: ±0.004) sunt extrem de stabile
  - Categorii "grele" (Action: ±0.022, Evidence: ±0.017) variază mai mult — performanța lor depinde semnificativ de eșantionul de antrenare

### 3.3 Analiza erorilor (per categorie, exp01a_full pe val seed=42)

Distribuția cazurilor din val_split (863 sample-uri):

| Marker | TP | FP | FN_partial | FN_missed | Gold total | Pred total |
|---|---|---|---|---|---|---|
| Action | 237 | 774 | 133 | 547 | 917 | 1011 |
| Actor | 544 | 673 | 58 | 620 | 1222 | 1217 |
| Effect | 164 | 568 | 103 | 437 | 704 | 732 |
| Evidence | 159 | 546 | 83 | 443 | 685 | 705 |
| Victim | 193 | 345 | 22 | 381 | 596 | 538 |

**Legendă:**
- **TP** = True Positive (IoU ≥ 0.5 cu gold)
- **FP** = False Positive (predicție fără corespondent gold)
- **FN_partial** = gold ratat dar cu predicție apropiată (IoU > 0 dar < 0.5)
- **FN_missed** = gold complet ratat (nicio predicție apropiată, IoU = 0)

**Patternuri de erori:**

1. **Pentru categoriile abstracte (Action, Effect, Evidence):**
   - FP ~46% din total erori — modelul inventează frecvent span-uri inexistente
   - FN_missed ~33% — modelul omite complet ~⅓ din span-urile gold
   - Aceste categorii suferă de eroare **dublă**: prea generos + prea conservator simultan

2. **Pentru categoriile de tip entitate (Actor, Victim):**
   - FN_partial mic (2-3%) — predicțiile sunt fie clar corecte, fie clar greșite
   - Decizii mai binare, sugerând pattern matching mai net

3. **FN_missed >> FN_partial pe toate categoriile:**
   - 80% din erorile de tip FN sunt "omitere completă", nu "aproape"
   - Recall-ul limitat e cauzat mai mult de **eșec total al detecției** decât de imprecizie de margine

### 3.4 Lungimea span-urilor — sub-extragere sistematică

| Marker | Gold (chars) | Pred 1A | Pred 1B | Pred 1C |
|---|---|---|---|---|
| Action | **27.8** | 18.1 (-35%) | 16.3 (-41%) | 7.2 (-74%) |
| Actor | 12.4 | 10.1 (-19%) | 9.7 (-22%) | 6.8 (-45%) |
| Effect | **37.8** | 23.8 (-37%) | 22.8 (-40%) | 6.7 (-82%) |
| Evidence | **36.7** | 18.5 (-50%) | 16.0 (-56%) | 7.3 (-80%) |
| Victim | 12.9 | 9.6 (-26%) | 9.4 (-27%) | 6.9 (-46%) |

**Observații cheie:**
- Modelele învață să identifice **nucleul** unui span dar nu reușesc să extindă predicția la întregul context
- Sub-extragerea e mai pronunțată cu cât categoria are span-uri mai lungi (Effect, Evidence, Action)
- Linear probing produce span-uri de lungime aproape constantă (6.7-7.3 chars) indiferent de categorie — sugerează că reprezentările pre-antrenate **nu codifică intrinsec noțiunea de span coerent**

## 4. Interpretări și insight-uri

### 4.1 Ierarhia dificultății categoriilor

Pe baza F1 Full FT:

1. **Actor (0.436)** — cea mai ușoară. Entități nominale cu features lexicale clare (nume proprii, capitalize, organizații).
2. **Victim (0.330)** — moderată. Tot entități nominale, dar definirea cere context (cine suferă acțiunea).
3. **Action (0.232) ≈ Effect (0.237) ≈ Evidence (0.223)** — cele mai grele. Concepte abstracte, contextuale, cu span-uri lungi și variabile.

Această ierarhie corelează cu **complexitatea conceptuală**: cu cât categoria e mai abstractă, cu atât DistilBERT pierde mai mult.

### 4.2 Linear probing eșuează catastrofal pe categoriile grele

Recall pe linear probe:
- Action: 1.96% (875 missed din 893 gold)
- Effect: 0.28% (2/693 detectate)
- Evidence: 0.29% (2/676 detectate)

**Concluzie:** reprezentările pre-antrenate ale DistilBERT-base-uncased **nu conțin features lineare separabile** pentru aceste concepte abstracte. Acest rezultat motivează necesitatea fine-tuning-ului profund — sau a modelelor cu reprezentări mai bogate (LLM-uri moderne, abordare investigată în Etapa 2).

### 4.3 Saltul uniform Unfreeze 4 → Full FT

Diferența F1 Full vs Unfreeze 4 e remarcabil de constantă (0.033-0.044) pe toate categoriile. Asta sugerează că ultimele 2 layere transformer (cele pe care Unfreeze 4 le îngheață) fac un **"general task adaptation"** mai degrabă decât specializare pe categorie. Layere mai timpurii, când sunt fine-tunate, beneficiază uniform toate categoriile.

### 4.4 Tipologia textului afectează performanța

Pe baza analizei calitative, textele descriptive cu Effect explicit menționat ca propoziție unitară (ex: articole de știri) duc la predicții corecte cu IoU=1.0. Pe texte argumentative, subiective (ex: eseuri Reddit), modelul produce span-uri scurte și incoerente. **Domeniul textului e o variabilă semnificativă** care nu e capturată în metrica agregată.

### 4.5 Confuzia de roluri Actor ↔ Victim

Exemplu real:
- Text: "...what's taking place in Bolivia, referencing some similar US backed coups..."
- Gold Actor: "US"
- Pred Actor (FP): "Bolivia"

Modelul confundă rolurile când ambele entități au caracteristici lexicale similare. Distincția cere înțelegere a **relației cauzale** dintre Actor și Victim — capabilitate slabă în DistilBERT.

### 4.6 Limitări de tokenizare WordPiece

Exemplu real:
- Gold Victim: "unvaccinated"
- Pred Victim: "vaccina"

WordPiece fragmentează "unvaccinated" → `un + ##vaccina + ##ted`. Modelul învață să marcheze sub-tokenul de bază (`##vaccina`), pierzând prefixele negative (`un-`) care **inversează sensul**. Aceasta e o limitare arhitecturală pe care vocabularele BPE moderne (ex: Qwen, Llama) ar putea-o ameliora.

### 4.7 Shortcut learning pe Evidence

Exemplu real:
- FP Evidence: "to a report" (din "according to a report")

Modelul activează pe markeri meta-textuali ('according to', 'These surveys') în loc de conținutul evidenței propriu-zise. **Învață colocații sintagmatice fără înțelegere semantică** — un pattern clasic de shortcut learning.

## 5. Concluziile Etapei 1

1. **Metoda DistilBERT cu schemă one-vs-rest produce baseline reproductibil** (F1 micro = 0.305 ± 0.007 pe Full FT) consistent cu rezultatele oficiale SemEval.

2. **Stratificarea fine-tuning-ului confirmă ipotezele transfer learning:** linear probing eșuează (0.055 macro), partial fine-tuning recuperează majoritar (0.253), full fine-tuning adaugă marginal (0.292).

3. **Categoriile abstracte (Action, Effect, Evidence) rămân mult sub potențial** (~0.22-0.24 F1) — limitarea principală a DistilBERT pentru această sarcină.

4. **Erorile dominante** sunt omiterile complete (FN_missed) și sub-extragerea sistematică, sugerând limitări de capacitate semantică, nu doar de margine de span.

5. **Argument natural pentru Etapa 2** (LLM modern + LoRA): modelele cu reprezentări mai bogate ar putea ameliora exact categoriile unde DistilBERT eșuează.

## 6. Fișiere și artefacte

### 6.1 Cod

| Fișier | Descriere |
|---|---|
| `train_one_span.py` | Antrenare per categorie (one-vs-rest) cu argparse complet |
| `infer_one_span.py` | Inferență pe test set, generează submission JSONL |
| `eval_token.py` | Evaluator oficial SemEval (token-based F1 cu IoU≥0.5) |
| `qualitative_analysis.py` | Analiza erorilor: TP/FP/FN_partial/FN_missed + exemple |

### 6.2 Date

| Fișier | Descriere |
|---|---|
| `data/rehydrated/train_rehydrated.jsonl` | 4316 sample-uri cu text complet |
| `data/rehydrated/val_split.jsonl` | 863 sample-uri val (seed=42) |
| `data/rehydrated/val_split_seed123.jsonl` | val seed=123 (multi-seed) |
| `data/rehydrated/val_split_seed2024.jsonl` | val seed=2024 (multi-seed) |
| `data/rehydrated/dev_rehydrated.jsonl` | 100 sample-uri dev (gold ascuns, pentru submisia oficială) |

### 6.3 Modele antrenate (15 modele × 3 multi-seed = 25 total)

| Experiment | Marker dirs |
|---|---|
| `checkpoints/exp01a_full/` | Action, Actor, Effect, Evidence, Victim |
| `checkpoints/exp01b_unfreeze4/` | Action, Actor, Effect, Evidence, Victim |
| `checkpoints/exp01c_linear/` | Action, Actor, Effect, Evidence, Victim |
| `checkpoints/exp01a_full_seed123/` | Action, Actor, Effect, Evidence, Victim |
| `checkpoints/exp01a_full_seed2024/` | Action, Actor, Effect, Evidence, Victim |

### 6.4 Rezultate

| Fișier | Conținut |
|---|---|
| `results/etapa1_all_runs.csv` | Tabelul principal 3 strategii × 5 markeri (CSV pentru import în Word/Excel) |
| `results/etapa1_multiseed_1A.csv` | Multi-seed cu mean ± std |
| `results/qualitative/analysis_exp01a_full.json` | Statistici complete + 10 exemple per categorie per tip eroare |
| `results/qualitative/analysis_exp01b_unfreeze4.json` | Idem pentru 1B |
| `results/qualitative/analysis_exp01c_linear.json` | Idem pentru 1C |
| `results/qualitative/exemple_pentru_lucrare.md` | Exemple curate selectate (anexă pentru lucrare) |

## 7. Pregătirea pentru Etapa 2

Insight-urile din Etapa 1 sugerează direcții concrete pentru Etapa 2 (LoRA pe LLM modern):

1. **Modele candidate:** Qwen2.5-0.5B (primary), Llama-3.2-1B (alternative). Vocabular BPE mai larg, antrenament mai recent, posibil reprezentări mai bune pentru concepte abstracte.

2. **Așteptări de îmbunătățire:**
   - Action, Effect, Evidence — categorii unde DistilBERT eșuează cel mai mult — au cel mai mare potențial de creștere
   - Actor, Victim — deja relativ bine prinse — probabil creșteri modeste

3. **Ipoteze de testat:**
   - LLM-urile moderne reduc shortcut learning pe Evidence
   - Vocabular BPE rezolvă problemele de tokenizare (ex. "unvaccinated")
   - Capacitatea contextuală mai mare reduce sub-extragerea pe span-uri lungi

4. **Setup experimental Etapa 2:**
   - LoRA cu r=16, target_modules=[q_proj, v_proj]
   - Aceeași schemă one-vs-rest pe 5 markeri
   - Multi-seed (42, 123, 2024) pentru reproductibilitate
   - Ablație rang LoRA (r ∈ {4, 8, 16, 32}) pe Victim
   - Comparație directă cu Etapa 1 pe metrica F1 token-based

---

**Status documentare Etapa 1:** ✅ COMPLET
**Următor:** Etapa 2 — LoRA pe LLM modern (Qwen2.5-0.5B)
