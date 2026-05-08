"""
train_one_span.py — versiune modificata pentru lucrarea de licenta
SemEval 2026 Task 10 (PsyCoMark) — Span Extraction Subtask

Compatibil cu transformers >= 4.57.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(file_path: str) -> list:
    data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()[:80]}...")
    return data


def split_train_val(data: list, val_ratio: float, seed: int):
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_val = int(len(data) * val_ratio)
    val_idx = set(indices[:n_val])
    train = [data[i] for i in range(len(data)) if i not in val_idx]
    val = [data[i] for i in range(len(data)) if i in val_idx]
    return train, val


def create_label_maps_simplified(marker_type: str):
    label_list = ["O", marker_type]
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    return label_to_id, id_to_label, len(label_list)


def tokenize_and_align_labels_simplified(examples, tokenizer, label_to_id, marker_type, max_length):
    """
    Tokenii speciali ([CLS], [SEP], [PAD]) primesc -100 (ignorati de loss).
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
    )

    labels = []
    all_markers = examples.get("markers", [])
    marker_label_id = label_to_id[marker_type]

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        example_labels = []
        for start, end in offsets:
            if start is None or end is None or (start == 0 and end == 0):
                example_labels.append(-100)
            else:
                example_labels.append(0)

        example_markers = all_markers[i] if i < len(all_markers) else []
        for marker in example_markers:
            if marker["type"] != marker_type:
                continue
            start_char = marker["startIndex"]
            end_char = marker["endIndex"]
            for token_idx, (start, end) in enumerate(offsets):
                if start is None or end is None:
                    continue
                if start == 0 and end == 0:
                    continue
                if start_char <= start < end_char or (start < end_char and end > start_char):
                    if token_idx < len(example_labels) and example_labels[token_idx] == 0:
                        example_labels[token_idx] = marker_label_id

        labels.append(example_labels)

    tokenized["labels"] = labels
    tokenized.pop("offset_mapping", None)
    return tokenized


def make_compute_metrics(num_labels: int):
    from sklearn.metrics import precision_recall_fscore_support

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        flat_preds = preds.flatten()
        flat_labels = labels.flatten()
        mask = flat_labels != -100
        flat_preds = flat_preds[mask]
        flat_labels = flat_labels[mask]

        precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(
            flat_labels, flat_preds, labels=[1], average="binary", zero_division=0
        )
        _, _, f1_micro, _ = precision_recall_fscore_support(
            flat_labels, flat_preds, average="micro", zero_division=0
        )
        _, _, f1_macro, _ = precision_recall_fscore_support(
            flat_labels, flat_preds, average="macro", zero_division=0
        )

        return {
            "precision": float(precision_pos),
            "recall": float(recall_pos),
            "f1": float(f1_pos),
            "f1_micro": float(f1_micro),
            "f1_macro": float(f1_macro),
        }

    return compute_metrics


def configure_trainable_layers(model, unfreeze_last_n: int) -> None:
    """
    DistilBertForTokenClassification:
      - distilbert.embeddings
      - distilbert.transformer.layer  (lista de 6)
      - dropout
      - classifier  (Linear: 768 -> num_labels)

    NU exista pre_classifier (acela e doar la SequenceClassification).

    unfreeze_last_n=0 -> doar capul (linear probing)
    unfreeze_last_n=4 -> ultimele 4 layere + cap
    unfreeze_last_n=6 -> tot DistilBERT (full fine-tuning)
    """
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    transformer_layers = model.distilbert.transformer.layer
    total_layers = len(transformer_layers)
    if unfreeze_last_n > total_layers:
        print(f"  WARNING: unfreeze_last_n={unfreeze_last_n} > total layers ({total_layers}), capping.")
        unfreeze_last_n = total_layers

    for i in range(total_layers - unfreeze_last_n, total_layers):
        for param in transformer_layers[i].parameters():
            param.requires_grad = True

    if unfreeze_last_n >= total_layers:
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total
    print(f"  Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    print(f"  Strategy: unfreeze_last_n={unfreeze_last_n} of {total_layers} transformer layers")


def parse_args():
    p = argparse.ArgumentParser(description="Train DistilBERT span extractor (one-vs-rest per marker type)")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--marker_types", type=str, nargs="+",
                   default=["Action", "Actor", "Effect", "Evidence", "Victim"])
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--unfreeze_last_n", type=int, default=6,
                   help="Cate layere transformer (din 6) sunt antrenate. 0=linear probe, 6=full.")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Args: {vars(args)}")
    print(f"\nLoading data from {args.data_path}...")
    all_data = load_data(args.data_path)
    train_data, val_data = split_train_val(all_data, args.val_ratio, args.seed)
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)

    all_metrics = {}

    for marker_type in args.marker_types:
        print(f"\n{'=' * 70}")
        print(f"Training for marker type: {marker_type}")
        print(f"{'=' * 70}")

        label_to_id, id_to_label, num_labels = create_label_maps_simplified(marker_type)

        tokenize_kwargs = {
            "tokenizer": tokenizer,
            "label_to_id": label_to_id,
            "marker_type": marker_type,
            "max_length": args.max_length,
        }
        tokenized_train = train_dataset.map(
            tokenize_and_align_labels_simplified, batched=True, fn_kwargs=tokenize_kwargs,
            remove_columns=train_dataset.column_names,
        )
        tokenized_val = val_dataset.map(
            tokenize_and_align_labels_simplified, batched=True, fn_kwargs=tokenize_kwargs,
            remove_columns=val_dataset.column_names,
        )

        model = DistilBertForTokenClassification.from_pretrained(
            args.model_name, num_labels=num_labels,
            id2label=id_to_label, label2id=label_to_id,
        )

        configure_trainable_layers(model, args.unfreeze_last_n)

        marker_output_dir = os.path.join(args.output_dir, marker_type)
        os.makedirs(marker_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=marker_output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="linear",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=50,
            report_to="none",
            seed=args.seed,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)
        compute_metrics = make_compute_metrics(num_labels)

        # IMPORTANT: in transformers 4.46+, parametrul se numeste
        # processing_class, NU mai e tokenizer.
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        )

        train_result = trainer.train()
        print(f"\nTraining for {marker_type} finished. Final train loss: {train_result.training_loss:.4f}")

        eval_metrics = trainer.evaluate()
        print(f"Best val metrics for {marker_type}: {eval_metrics}")

        final_model_dir = os.path.join(marker_output_dir, "final_model")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        print(f"Saved final model to: {final_model_dir}")

        all_metrics[marker_type] = {
            "best_val_f1": eval_metrics.get("eval_f1", 0.0),
            "best_val_precision": eval_metrics.get("eval_precision", 0.0),
            "best_val_recall": eval_metrics.get("eval_recall", 0.0),
            "best_val_f1_micro": eval_metrics.get("eval_f1_micro", 0.0),
            "best_val_f1_macro": eval_metrics.get("eval_f1_macro", 0.0),
            "final_train_loss": train_result.training_loss,
            "epochs_trained": train_result.metrics.get("epoch", args.num_epochs),
        }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"args": vars(args), "metrics": all_metrics}, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"All training done. Metrics saved to: {metrics_path}")
    print(f"{'=' * 70}")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
