"""
qualitative_analysis.py — Analiza calitativă a predicțiilor DistilBERT pe span extraction.

Pentru fiecare experiment (1A/1B/1C) și categorie, scriptul:
1. Calculează statistici globale (TP, FP, FN, lungime medie span)
2. Extrage exemple representative:
   - True Positives (corecte)
   - False Negatives (ratate)
   - False Positives (inventate)
   - Partial overlaps (aproape, dar IoU < 0.5)
3. Salvează totul ca JSON pentru a-l avea disponibil pentru lucrare.

Span match folosește IoU pe tokeni (white-space tokenization) cu prag 0.5,
identic cu eval_token.py oficial.
"""

import argparse
import json
import os
from collections import defaultdict


# ---------------------------------------------------------------------------
# IoU pe tokeni (compatibil cu eval_token.py oficial)
# ---------------------------------------------------------------------------

def text_to_token_offsets(text):
    """Tokenizare pe whitespace, cu offset-uri caracter."""
    tokens = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        start = i
        while i < len(text) and not text[i].isspace():
            i += 1
        tokens.append((start, i))  # [start_char, end_char)
    return tokens


def char_span_to_token_span(start_char, end_char, token_offsets):
    """Convertește un span (char_start, char_end) la (tok_start, tok_end)."""
    tok_start = None
    tok_end = None
    for tok_idx, (ts, te) in enumerate(token_offsets):
        # Token e în span dacă există overlap
        if te <= start_char:
            continue
        if ts >= end_char:
            break
        if tok_start is None:
            tok_start = tok_idx
        tok_end = tok_idx + 1
    return tok_start, tok_end


def iou_token_span(s1, s2):
    """IoU între două (tok_start, tok_end) intervals."""
    if s1 is None or s2 is None:
        return 0.0
    a_start, a_end = s1
    b_start, b_end = s2
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# Match span gold ↔ predict
# ---------------------------------------------------------------------------

def match_spans(gold_spans, pred_spans, text, iou_threshold=0.5):
    """
    Pentru o singură pereche (gold list, pred list) pe o singură categorie:
    returnează liste de (gold, best_pred, iou) pentru fiecare gold,
    și un set de pred-uri folosite.

    O potrivire e considerată match dacă IoU >= threshold.
    """
    token_offsets = text_to_token_offsets(text)

    gold_token_spans = []
    for g in gold_spans:
        ts = char_span_to_token_span(g['startIndex'], g['endIndex'], token_offsets)
        gold_token_spans.append(ts)

    pred_token_spans = []
    for p in pred_spans:
        ts = char_span_to_token_span(p['startIndex'], p['endIndex'], token_offsets)
        pred_token_spans.append(ts)

    # Greedy match: pentru fiecare gold, găsește predict cu IoU maxim
    matches = []  # listă de dict-uri
    used_pred = set()

    for gi, (g, gts) in enumerate(zip(gold_spans, gold_token_spans)):
        best_iou = 0.0
        best_pi = -1
        for pi, (p, pts) in enumerate(zip(pred_spans, pred_token_spans)):
            if pi in used_pred:
                continue
            iou = iou_token_span(gts, pts)
            if iou > best_iou:
                best_iou = iou
                best_pi = pi
        if best_pi >= 0 and best_iou >= iou_threshold:
            used_pred.add(best_pi)
            matches.append({
                "type": "TP",  # match peste prag
                "gold": g,
                "pred": pred_spans[best_pi],
                "iou": round(best_iou, 3),
            })
        elif best_pi >= 0 and best_iou > 0:
            # Există overlap dar sub prag
            matches.append({
                "type": "FN_partial",  # gold neacoperit, există pred apropiat
                "gold": g,
                "pred": pred_spans[best_pi],
                "iou": round(best_iou, 3),
            })
        else:
            matches.append({
                "type": "FN_missed",  # gold complet ratat (nicio predicție apropiată)
                "gold": g,
                "pred": None,
                "iou": 0.0,
            })

    # Pred-uri rămase = false positives
    fps = []
    for pi, p in enumerate(pred_spans):
        if pi not in used_pred:
            fps.append({"type": "FP", "gold": None, "pred": p, "iou": 0.0})

    return matches + fps


# ---------------------------------------------------------------------------
# Analiza pe un întreg experiment
# ---------------------------------------------------------------------------

def analyze_experiment(gold_file, pred_file, marker_types, iou_threshold=0.5,
                       num_examples_per_category=5):
    """
    Returnează un dict cu:
    - statistici per categorie (TP, FP, FN_partial, FN_missed, span_lengths)
    - exemple per categorie per tip
    """
    # Încarcă gold
    gold_by_id = {}
    with open(gold_file) as f:
        for line in f:
            ex = json.loads(line)
            gold_by_id[ex['_id']] = ex

    # Încarcă predicțiile
    pred_by_id = {}
    with open(pred_file) as f:
        for line in f:
            ex = json.loads(line)
            pred_by_id[ex['_id']] = ex

    # Inițializare structură pentru rezultate
    results = {
        marker: {
            "stats": {
                "TP": 0, "FP": 0, "FN_partial": 0, "FN_missed": 0,
                "gold_total": 0, "pred_total": 0,
                "gold_span_lengths_chars": [],
                "pred_span_lengths_chars": [],
            },
            "examples": {
                "TP": [], "FP": [], "FN_partial": [], "FN_missed": [],
            }
        }
        for marker in marker_types
    }

    # Iterare peste sample-uri
    for sample_id, gold_ex in gold_by_id.items():
        pred_ex = pred_by_id.get(sample_id)
        if pred_ex is None:
            continue

        text = gold_ex.get('text', '')
        gold_markers = gold_ex.get('markers') or []
        pred_markers = pred_ex.get('markers') or []

        for marker in marker_types:
            gold_for_marker = [m for m in gold_markers if m.get('type') == marker]
            pred_for_marker = [m for m in pred_markers if m.get('type') == marker]

            results[marker]["stats"]["gold_total"] += len(gold_for_marker)
            results[marker]["stats"]["pred_total"] += len(pred_for_marker)

            for g in gold_for_marker:
                results[marker]["stats"]["gold_span_lengths_chars"].append(
                    g['endIndex'] - g['startIndex']
                )
            for p in pred_for_marker:
                results[marker]["stats"]["pred_span_lengths_chars"].append(
                    p['endIndex'] - p['startIndex']
                )

            matches = match_spans(gold_for_marker, pred_for_marker, text, iou_threshold)
            for m in matches:
                m_type = m["type"]
                results[marker]["stats"][m_type] += 1
                # Salvăm exemplele cu context
                if len(results[marker]["examples"][m_type]) < num_examples_per_category:
                    example_obj = {
                        "sample_id": sample_id,
                        "text_excerpt": text[:300] + "..." if len(text) > 300 else text,
                        "gold_span": m.get("gold"),
                        "pred_span": m.get("pred"),
                        "iou": m.get("iou"),
                    }
                    results[marker]["examples"][m_type].append(example_obj)

    # Calcule finale: precision, recall, F1, lungimi medii
    summary = {}
    for marker in marker_types:
        s = results[marker]["stats"]
        tp = s["TP"]
        fp = s["FP"]
        fn = s["FN_partial"] + s["FN_missed"]  # toate gold-urile neacoperite
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        gold_lens = s["gold_span_lengths_chars"]
        pred_lens = s["pred_span_lengths_chars"]
        avg_gold_len = sum(gold_lens) / len(gold_lens) if gold_lens else 0.0
        avg_pred_len = sum(pred_lens) / len(pred_lens) if pred_lens else 0.0

        summary[marker] = {
            "TP": tp, "FP": fp,
            "FN_partial": s["FN_partial"], "FN_missed": s["FN_missed"],
            "gold_total": s["gold_total"], "pred_total": s["pred_total"],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "avg_gold_span_len_chars": round(avg_gold_len, 1),
            "avg_pred_span_len_chars": round(avg_pred_len, 1),
        }

    return {
        "summary": summary,
        "examples": {marker: results[marker]["examples"] for marker in marker_types},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gold_file", required=True, help="val_split.jsonl cu gold-standard")
    p.add_argument("--pred_dir", required=True, help="folder cu submission-urile (cu val_submission_full.jsonl)")
    p.add_argument("--output_dir", required=True, help="folder pentru output (analysis_<exp>.json)")
    p.add_argument("--experiments", nargs="+",
                   default=["exp01a_full", "exp01b_unfreeze4", "exp01c_linear"])
    p.add_argument("--marker_types", nargs="+",
                   default=["Action", "Actor", "Effect", "Evidence", "Victim"])
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--num_examples", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for exp in args.experiments:
        pred_file = os.path.join(args.pred_dir, exp, "val_submission_full.jsonl")
        if not os.path.exists(pred_file):
            print(f"  SKIP {exp}: nu există {pred_file}")
            continue

        print(f"\n{'=' * 70}")
        print(f"Analiză: {exp}")
        print(f"{'=' * 70}")

        analysis = analyze_experiment(
            gold_file=args.gold_file,
            pred_file=pred_file,
            marker_types=args.marker_types,
            iou_threshold=args.iou_threshold,
            num_examples_per_category=args.num_examples,
        )

        # Print summary
        print(f"\n{'Marker':<10} {'TP':<5} {'FP':<5} {'FN_p':<6} {'FN_m':<6} "
              f"{'P':<8} {'R':<8} {'F1':<8} {'GoldLen':<8} {'PredLen':<8}")
        for marker in args.marker_types:
            s = analysis["summary"][marker]
            print(f"{marker:<10} {s['TP']:<5} {s['FP']:<5} {s['FN_partial']:<6} {s['FN_missed']:<6} "
                  f"{s['precision']:<8.4f} {s['recall']:<8.4f} {s['f1']:<8.4f} "
                  f"{s['avg_gold_span_len_chars']:<8.1f} {s['avg_pred_span_len_chars']:<8.1f}")

        # Salvare
        out_path = os.path.join(args.output_dir, f"analysis_{exp}.json")
        with open(out_path, "w") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
