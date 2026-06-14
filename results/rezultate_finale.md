# Rezultate finale

Metrica oficială: F1 la nivel de token, cu prag de suprapunere IoU ≥ 0,5. Fiecare configurație principală a fost rulată de cinci ori, fără *seed* fixat; se raportează media și deviația standard. Evaluarea s-a făcut pe partiția internă de validare (863 exemple).

## Etapa 1 — DistilBERT

### Strategii de *fine-tuning*

| Strategie | Parametri antrenabili | F1 macro | Timp antrenare / marker |
| --- | --- | --- | --- |
| Linear probing | 0,15 mil. (0,2%) | 0,055 | ~1,7 min |
| Unfreeze 4 straturi | 21,30 mil. (32%) | 0,253 | ~7,2 min |
| Fine-tuning complet | 66,90 mil. (100%) | **0,2916 ± 0,0079** | ~9,2 min |

*Linear probing și unfreeze 4 sunt rulări exploratorii unice (folosite pentru a stabili ierarhia strategiilor); fine-tuning-ul complet a fost evaluat pe 5 rulări.*

### Fine-tuning complet, pe categorii (5 rulări)

| Marker | F1 token IoU ≥ 0,5 |
| --- | --- |
| Action | 0,2388 ± 0,0077 |
| Actor | 0,4311 ± 0,0141 |
| Effect | 0,2168 ± 0,0135 |
| Evidence | 0,2413 ± 0,0223 |
| Victim | 0,3303 ± 0,0171 |
| **F1 Macro** | **0,2916 ± 0,0079** |
| F1 Micro | 0,3053 ± 0,0083 |

## Etapa 2 — Qwen2.5-0.5B + LoRA

### Rezultate pe categorii (5 rulări)

| Marker | F1 token IoU ≥ 0,5 |
| --- | --- |
| Action | 0,2534 ± 0,0205 |
| Actor | 0,4092 ± 0,0184 |
| Effect | 0,2211 ± 0,0227 |
| Evidence | 0,2328 ± 0,0183 |
| Victim | 0,3483 ± 0,0209 |
| **F1 Macro** | **0,2930 ± 0,0149** |
| F1 Micro | 0,3048 ± 0,0135 |

## Comparație directă (medii pe 5 rulări)

| Marker | DistilBERT *fine-tuning* | Qwen + LoRA | ∆ |
| --- | --- | --- | --- |
| Action | 0,2388 | 0,2534 | +0,0146 |
| Actor | 0,4311 | 0,4092 | −0,0219 |
| Effect | 0,2168 | 0,2211 | +0,0043 |
| Evidence | 0,2413 | 0,2328 | −0,0085 |
| Victim | 0,3303 | 0,3483 | +0,0180 |
| **F1 Macro** | 0,2916 | 0,2930 | **+0,0014** |
| F1 Micro | 0,3053 | 0,3048 | −0,0005 |

Diferența la F1 macro (0,0014) este sub deviația standard a fiecărei configurații, deci cele două abordări sunt echivalente statistic — cu o reducere de ~15× a numărului de parametri actualizați în cazul LoRA.

## Ablații

### Ponderarea clasei pozitive (LoRA, 3 rulări)

| Configurație | F1 macro | Stabilitate |
| --- | --- | --- |
| Cu ponderare (3,0) | 0,2930 ± 0,0149 | toate rulările stabile |
| Fără ponderare (1,0) | 0,2344 ± 0,1131 | colaps la 0 pe categorii întregi în 1 din 3 rulări |

Fără ponderare, antrenarea devine instabilă: într-una dintre rulări, trei categorii (Actor, Effect, Evidence) au colapsat complet. Deviația standard de ~7× mai mare confirmă utilitatea ponderii.

### Contribuția LoRA (3 rulări)

| Configurație Qwen2.5-0.5B | Parametri antrenabili | F1 macro | Timp / marker |
| --- | --- | --- | --- |
| Doar capul (fără LoRA) | ~1,8 mii | 0,1197 ± 0,0048 | 11,0 min |
| LoRA (r = 32) | ~4,3 mil. | 0,2930 ± 0,0149 | 17,6 min |

Doar capul de clasificare peste modelul înghețat scade sub baseline-ul oficial (0,15); LoRA aproape dublează scorul. LoRA nu reduce timpul de antrenare (îl crește față de varianta doar-cap), avantajul fiind numărul redus de parametri actualizați și amprenta de memorie mai mică.

### Rangul LoRA (marker Victim, o rulare)

| Rang (r) | Alpha | Parametri antrenabili | F1 Victim |
| --- | --- | --- | --- |
| 4 | 8 | 544.260 (0,11%) | 0,2845 |
| 32 | 64 | 4.328.964 (0,87%) | 0,3302 |
| 64 | 128 | 8.654.340 (1,72%) | 0,3345 |

Saltul mare 4 → 32 (+0,046) și platoul 32 → 64 (+0,004) justifică alegerea r = 32 ca echilibru între capacitate și eficiență.

## Referință

Baseline oficial (pachet de start): ~0,15 F1 *overlap* pe setul de dev.
