"""
infer_one_span.py — versiune modificata pentru lucrarea de licenta
SemEval 2026 Task 10 (PsyCoMark) — Span Extraction Subtask

Modificari fata de versiunea oficiala:
  - argparse complet
  - cauta modelul salvat explicit (final_model/) inainte de checkpoints
  - max_length configurabil (sa fie consistent cu antrenarea)
  - print explicit al caii incarcate (sanity check)
  - foloseste AutoModel/AutoTokenizer pentru flexibilitate (Etapa 2 cu LLM)
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)


# ---------------------------------------------------------------------------
# Localizare model salvat
# ---------------------------------------------------------------------------

def resolve_model_path(model_dir: str, marker_type: str) -> str:
    """
    Cauta modelul antrenat pentru marker_type in urmatoarea ordine:
      1. {model_dir}/{marker_type}/final_model/   (salvat explicit)
      2. {model_dir}/{marker_type}/checkpoint-XXX (cel mai mare numar)
      3. {model_dir}/{marker_type}/                (fallback)

    Aceasta ordine asigura ca folosim modelul SALVAT EXPLICIT (best model
    dupa load_best_model_at_end), nu ultimul checkpoint care poate fi mai
    rau decat best.
    """
    base = os.path.join(model_dir, marker_type)

    # Optiunea 1: final_model salvat explicit
    final_path = os.path.join(base, "final_model")
    if os.path.isdir(final_path) and os.path.exists(os.path.join(final_path, "config.json")):
        print(f"  [load] final_model: {final_path}")
        return final_path

    # Optiunea 2: cel mai mare checkpoint
    if os.path.isdir(base):
        checkpoints = []
        for name in os.listdir(base):
            if name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-")[1])
                    checkpoints.append((step, os.path.join(base, name)))
                except (IndexError, ValueError):
                    continue
        if checkpoints:
            checkpoints.sort()
            latest = checkpoints[-1][1]
            print(f"  [load] latest checkpoint: {latest}")
            return latest

    # Optiunea 3: folderul de baza
    if os.path.isdir(base) and os.path.exists(os.path.join(base, "config.json")):
        print(f"  [load] base folder: {base}")
        return base

    raise FileNotFoundError(f"No model found for marker_type={marker_type} in {base}")


# ---------------------------------------------------------------------------
# Incarcare date (pastrata din versiunea oficiala)
# ---------------------------------------------------------------------------

def load_data(file_path: str) -> list:
    data = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                item["_id"] = item.get("_id", f"sample_{i}")
                item["text"] = item.get("text", "")
                item["markers"] = item.get("markers", [])
                item["conspiracy"] = item.get("conspiracy", "No")
                data.append(item)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line at {i}: {line.strip()[:80]}")
    print(f"Loaded {len(data)} samples for inference.")
    return data


def tokenize_for_inference(examples, tokenizer, max_length):
    """Tokenizeaza textul, pastreaza offset_mapping pentru reconstructia spanurilor."""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
    )
    # Trainer.predict cere o cheie 'labels'; o populam cu -100 (ignorata)
    tokenized["labels"] = [[-100] * len(om) for om in tokenized["offset_mapping"]]
    return tokenized


# ---------------------------------------------------------------------------
# Reconstructie spanuri (pastrata din versiunea oficiala, cu mici curatari)
# ---------------------------------------------------------------------------

def reconstruct_spans(predictions, tokenized_dataset, marker_type: str):
    reconstructed = defaultdict(list)

    for i, pred_ids in enumerate(predictions):
        offsets = tokenized_dataset[i]["offset_mapping"]
        original_text = tokenized_dataset[i]["text"]

        current_start = None

        def is_special(offset_tuple):
            if offset_tuple is None:
                return True
            s, e = offset_tuple
            if s is None or e is None:
                return True
            if s == 0 and e == 0:
                return True
            return False

        for token_idx, label_id in enumerate(pred_ids):
            offset = offsets[token_idx]

            if is_special(offset):
                # inchide span deschis (daca exista)
                if current_start is not None:
                    prev_end = None
                    if token_idx > 0 and not is_special(offsets[token_idx - 1]):
                        prev_end = offsets[token_idx - 1][1]
                    if prev_end is not None:
                        reconstructed[i].append({
                            "startIndex": current_start,
                            "endIndex": prev_end,
                            "type": marker_type,
                            "text": original_text[current_start:prev_end],
                        })
                    current_start = None
                continue

            start_char = offset[0]

            if label_id == 1:  # marker
                if current_start is None:
                    current_start = start_char
            else:  # "O"
                if current_start is not None:
                    prev_end = offsets[token_idx - 1][1] if token_idx > 0 else start_char
                    reconstructed[i].append({
                        "startIndex": current_start,
                        "endIndex": prev_end,
                        "type": marker_type,
                        "text": original_text[current_start:prev_end],
                    })
                    current_start = None

        # span deschis la final?
        if current_start is not None:
            last_end = None
            for k in range(len(pred_ids) - 1, -1, -1):
                if not is_special(offsets[k]):
                    last_end = offsets[k][1]
                    break
            if last_end is not None:
                reconstructed[i].append({
                    "startIndex": current_start,
                    "endIndex": last_end,
                    "type": marker_type,
                    "text": original_text[current_start:last_end],
                })

    return reconstructed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Inference for span extraction (one-vs-rest)")
    p.add_argument("--model_dir", type=str, required=True,
                   help="Folder cu sub-foldere per marker_type (ex: distilbert-exp01a)")
    p.add_argument("--test_file", type=str, required=True,
                   help="JSONL de evaluare (dev_rehydrated.jsonl)")
    p.add_argument("--submission_file", type=str, required=True,
                   help="Output: submission JSONL")
    p.add_argument("--marker_types", type=str, nargs="+",
                   default=["Action", "Actor", "Effect", "Evidence", "Victim"])
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--tokenizer_name", type=str, default="distilbert-base-uncased",
                   help="Folosit doar daca tokenizer-ul nu e in folderul modelului")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Args: {vars(args)}")

    raw_data = load_data(args.test_file)
    if not raw_data:
        print("Error: no data loaded.")
        sys.exit(1)

    unique_ids = [d["_id"] for d in raw_data]
    conspiracy_keys = [d["conspiracy"] for d in raw_data]

    test_dataset = Dataset.from_list(raw_data)

    # Folosim tokenizer-ul incarcat de la primul model gasit (ar trebui sa fie
    # acelasi pentru toate marker_types). Daca esueaza, fallback la default.
    try:
        first_path = resolve_model_path(args.model_dir, args.marker_types[0])
        tokenizer = AutoTokenizer.from_pretrained(first_path)
    except Exception:
        print(f"  Tokenizer fallback to: {args.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    tokenized_test = test_dataset.map(
        tokenize_for_inference, batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
        remove_columns=[c for c in test_dataset.column_names
                        if c not in ["text", "_id", "conspiracy"]],
    )

    all_predicted = defaultdict(list)

    for marker_type in args.marker_types:
        print(f"\n--- Inference for: {marker_type} ---")
        try:
            model_path = resolve_model_path(args.model_dir, marker_type)
            model = AutoModelForTokenClassification.from_pretrained(model_path)
        except Exception as e:
            print(f"  ERROR loading model for {marker_type}: {e}")
            continue

        data_collator = DataCollatorForTokenClassification(tokenizer)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"./tmp_inference_{marker_type}",
                per_device_eval_batch_size=args.batch_size,
                report_to="none",
            ),
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        pred_output = trainer.predict(tokenized_test)
        logits = pred_output.predictions
        pred_ids = np.argmax(logits, axis=2)

        markers_for_type = reconstruct_spans(pred_ids, tokenized_test, marker_type)
        for sample_idx, mks in markers_for_type.items():
            all_predicted[sample_idx].extend(mks)

        print(f"  Predicted {sum(len(v) for v in markers_for_type.values())} spans")

    # Scriere submission
    print(f"\nWriting submission to: {args.submission_file}")
    os.makedirs(os.path.dirname(args.submission_file) or ".", exist_ok=True)
    with open(args.submission_file, "w") as f:
        for i in range(len(raw_data)):
            obj = {
                "_id": unique_ids[i],
                "conspiracy": conspiracy_keys[i],
                "markers": all_predicted.get(i, []),
            }
            f.write(json.dumps(obj) + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
