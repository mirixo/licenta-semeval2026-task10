"""
infer_one_span_lora.py - Inferenta cu modele LoRA antrenate.

Adaptare a infer_one_span.py pentru:
- Model decoder-only Qwen2.5-0.5B
- Adapter LoRA in loc de model complet
- Tokenizare BPE Qwen

Genereaza submission.jsonl in formatul cerut de Codabench / eval_token.py:
{"id": "...", "markers": [{"type": "Action", "startIndex": X, "endIndex": Y}, ...]}
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForTokenClassification, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True,
                   help="folder cu modelele antrenate (subdir per marker)")
    p.add_argument("--test_file", required=True, help="fisier de test JSONL")
    p.add_argument("--submission_file", required=True,
                   help="fisier output (submission.jsonl)")
    p.add_argument("--marker_types", nargs="+", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B",
                   help="modelul de baza pe care a fost antrenat LoRA")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_model_for_marker(base_model_name, lora_adapter_dir, device):
    """Incarca modelul de baza si aplica adapter LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Incarca modelul de baza (Qwen) in FP32
    base_model = AutoModelForTokenClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    # Aplica adapter LoRA peste
    model = PeftModel.from_pretrained(base_model, lora_adapter_dir)
    model = model.merge_and_unload()  # Combina adapter cu base model pentru inferenta rapida
    model = model.to(device)
    model.eval()

    return model, tokenizer


def predict_spans_for_text(text, model, tokenizer, marker_type, max_length, device):
    """
    Aplica modelul pe un text si returneaza lista de span-uri prezise.

    Returneaza: list of dict {"type": marker_type, "startIndex": X, "endIndex": Y}
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    offsets = encoding["offset_mapping"][0].tolist()
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[0]  # [seq_len, 2]
    predictions = torch.argmax(logits, dim=-1).tolist()  # [seq_len]

    # Reconstruieste span-urile din token-uri consecutive cu eticheta 1
    spans = []
    current_start = None
    current_end = None

    for idx, (pred, (tok_start, tok_end)) in enumerate(zip(predictions, offsets)):
        # Token special (offset 0,0) - ignora
        if tok_start == tok_end:
            continue

        if pred == 1:
            if current_start is None:
                current_start = tok_start
                current_end = tok_end
            else:
                current_end = tok_end
        else:
            if current_start is not None:
                spans.append({
                    "type": marker_type,
                    "startIndex": int(current_start),
                    "endIndex": int(current_end),
                })
                current_start = None
                current_end = None

    # Daca textul se termina cu un span deschis
    if current_start is not None:
        spans.append({
            "type": marker_type,
            "startIndex": int(current_start),
            "endIndex": int(current_end),
        })

    return spans


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Incarca datele de test
    print(f"\nLoading test data from {args.test_file}")
    test_data = load_data(args.test_file)
    print(f"Total test samples: {len(test_data)}")

    # Initializeaza dictionarul de predictii indexat dupa _id
    predictions_by_id = {item["_id"]: {"_id": item["_id"], "markers": []}
                         for item in test_data}

    # Pentru fiecare marker, incarca modelul si genereaza predictii
    for marker in args.marker_types:
        print(f"\n{'=' * 70}")
        print(f"Marker: {marker}")
        print(f"{'=' * 70}")

        adapter_dir = os.path.join(args.model_dir, marker, "final_model")
        if not os.path.exists(adapter_dir):
            print(f"WARNING: adapter NOT FOUND at {adapter_dir} - skipping {marker}")
            continue

        print(f"Loading model from {adapter_dir}")
        model, tokenizer = load_model_for_marker(args.base_model, adapter_dir, device)

        # Aplica modelul pe fiecare sample
        for i, item in enumerate(test_data):
            text = item["text"]
            spans = predict_spans_for_text(
                text, model, tokenizer, marker, args.max_length, device
            )
            predictions_by_id[item["_id"]]["markers"].extend(spans)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(test_data)}")

        print(f"Done with {marker}")

        # Eliberare memorie
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Scrie submission file
    os.makedirs(os.path.dirname(args.submission_file), exist_ok=True)
    print(f"\nWriting submission to {args.submission_file}")
    with open(args.submission_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(predictions_by_id[item["_id"]]) + "\n")

    # Statistici
    total_spans = sum(len(p["markers"]) for p in predictions_by_id.values())
    print(f"\nTotal predicted spans: {total_spans}")
    print(f"Average spans per sample: {total_spans / len(test_data):.2f}")

    # Distributie per marker
    print("\nPredictions per marker:")
    for marker in args.marker_types:
        count = sum(1 for p in predictions_by_id.values()
                    for m in p["markers"] if m["type"] == marker)
        print(f"  {marker}: {count}")


if __name__ == "__main__":
    main()
