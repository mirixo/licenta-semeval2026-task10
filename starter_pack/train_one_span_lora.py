"""
train_one_span_lora.py - Antrenare LoRA pe Qwen2.5-0.5B pentru span extraction.

Adaptare a train_one_span.py din Etapa 1 pentru:
- Model decoder-only Qwen2.5-0.5B (in loc de DistilBERT encoder-only)
- Adaptare LoRA prin peft (in loc de unfreeze layers)
- Tokenizare BPE Qwen (in loc de WordPiece DistilBERT)

Schema ramane identica:
- One-vs-rest: un model independent per marker
- Etichete: 0=O, 1=I, -100=ignore
- Split 80/20 reproductibil cu seed
- Hugging Face Trainer + EarlyStopping
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Weighted Trainer (pentru class imbalance)
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """
    Trainer custom care foloseste class weights in CrossEntropyLoss.
    Util pentru categorii cu imbalance puternic (Effect, Evidence).
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight_tensor = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(
                weight=weight_tensor,
                ignore_index=-100,
            )
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # Flatten pentru CrossEntropyLoss
        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss



# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True, help="train_rehydrated.jsonl")
    p.add_argument("--output_dir", required=True, help="folder pentru modele")
    p.add_argument("--marker_types", nargs="+", required=True,
                   help="lista markerilor de antrenat")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B",
                   help="model HuggingFace base")
    p.add_argument("--lora_r", type=int, default=16, help="rang LoRA")
    p.add_argument("--lora_alpha", type=int, default=32, help="scaling alpha LoRA")
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--no_lora", action="store_true",
                   help="Daca e setat, NU aplica LoRA: ingheata modelul de baza, antreneaza doar capul de clasificare.")
    p.add_argument("--target_modules", nargs="+",
                   default=["q_proj", "v_proj"],
                   help="module pe care se aplica LoRA")
    p.add_argument("--num_epochs", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--train_file", type=str, default=None,
                   help="Daca e dat, se foloseste acest split in loc de split-ul intern.")
    p.add_argument("--val_file", type=str, default=None)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--class_weight_positive", type=float, default=1.0,
                   help="Greutate pentru clasa pozitiva (label 1) in loss. "
                        "Useful pentru class imbalance. Default 1.0 = fara weighting.")
    p.add_argument("--local_workdir", default=None,
                   help="Daca e setat, antrenarea se face local in acest folder "
                        "(rapid), apoi se copiaza doar final_model in output_dir.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading and split
# ---------------------------------------------------------------------------

def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def split_train_val(data, val_ratio):
    rng = random.Random()
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_val = int(len(data) * val_ratio)
    val_idx = set(indices[:n_val])
    train = [data[i] for i in range(len(data)) if i not in val_idx]
    val = [data[i] for i in range(len(data)) if i in val_idx]
    return train, val


# ---------------------------------------------------------------------------
# Label alignment cu BPE Qwen
# ---------------------------------------------------------------------------

def align_labels_to_tokens(examples, tokenizer, marker_type, max_length):
    """
    Tokenizeaza textele si aliniaza etichetele binare (O/I) la sub-tokeni
    folosind offset_mapping.

    Functioneaza identic ca pentru WordPiece - logica se bazeaza pe offset-uri,
    nu pe tipul tokenizer-ului.
    """
    texts = examples["text"]
    all_markers = examples["markers"]

    encodings = tokenizer(
        texts,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    labels_batch = []
    for idx, offsets in enumerate(encodings["offset_mapping"]):
        # Extrage span-urile gold pentru markerul curent
        markers = all_markers[idx] or []
        spans = [
            (m["startIndex"], m["endIndex"])
            for m in markers
            if m.get("type") == marker_type
        ]

        labels = []
        for tok_start, tok_end in offsets:
            # Token special ([CLS], [SEP], [PAD]) - offset (0,0) sau egal
            if tok_start == tok_end:
                labels.append(-100)
                continue

            # Verificam daca tokenul e in vreun span gold
            in_span = False
            for s, e in spans:
                if tok_start < e and tok_end > s:
                    in_span = True
                    break
            labels.append(1 if in_span else 0)

        labels_batch.append(labels)

    encodings["labels"] = labels_batch
    encodings.pop("offset_mapping")
    return encodings


# ---------------------------------------------------------------------------
# Metrici de evaluare
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """
    Calculeaza F1, Precision, Recall la nivel de token, ignorand -100.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    # Filtreaza tokens cu label -100
    true_labels = []
    pred_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_labels.append(l)
                pred_labels.append(p)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    tp = ((pred_labels == 1) & (true_labels == 1)).sum()
    fp = ((pred_labels == 1) & (true_labels == 0)).sum()
    fn = ((pred_labels == 0) & (true_labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ---------------------------------------------------------------------------
# Configurare model LoRA
# ---------------------------------------------------------------------------

def load_and_configure_model(model_name, lora_r, lora_alpha, lora_dropout,
                              target_modules, num_labels=2, no_lora=False):
    """
    Incarca modelul de baza si aplica LoRA.

    Returneaza modelul cu LoRA aplicat si tokenizer-ul.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Force FP32 pentru a permite FP16 mixed precision pe T4
    )

    # Aplica LoRA
    if no_lora:
        # Fara LoRA: ingheata tot modelul de baza, antreneaza doar capul de clasificare
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "score" in name or "classifier" in name:
                param.requires_grad = True
    else:
        # Aplica LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.TOKEN_CLS,
        )
        model = get_peft_model(model, lora_config)
        for name, param in model.named_parameters():
            if "score" in name or "classifier" in name:
                param.requires_grad = True

    return model, tokenizer

    # Verificam capul de classification e antrenabil
    # peft uneori il ingheata; il dezghetam explicit
    for name, param in model.named_parameters():
        if "score" in name or "classifier" in name:
            param.requires_grad = True

    return model, tokenizer


# ---------------------------------------------------------------------------
# Main loop per marker
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    run_entropy = int.from_bytes(os.urandom(4), "little")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"LoRA fine-tuning pe {args.model_name}")
    print(f"{'=' * 70}\n")

    # Incarca toate datele
    print(f"Loading data from {args.data_path}")
    all_data = load_data(args.data_path)
    print(f"Total samples: {len(all_data)}")

    # Split train/val reproductibil
    if args.train_file and args.val_file:
        # Split gata facut (toti markerii rularii impart aceeasi partitie)
        train_data = load_data(args.train_file)
        val_data = load_data(args.val_file)
        print(f"  Split primit din afara: train={len(train_data)} val={len(val_data)}")
    else:
        # Split intern, fara seed (comportamentul de pana acum)
        all_data = load_data(args.data_path)
        train_data, val_data = split_train_val(all_data, args.val_ratio)
        val_split_path = os.path.join(args.output_dir, "val_split_used.jsonl")
        with open(val_split_path, "w") as f:
            for ex in val_data:
                f.write(json.dumps(ex) + "\n")
        print(f"  Val split salvat: {val_split_path}")
        print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    # Antrenare per marker
    all_metrics = {}

    for marker in args.marker_types:
        print(f"\n{'=' * 70}")
        print(f"Marker: {marker}")
        print(f"{'=' * 70}")

        # Folder final pe Drive (unde se va copia modelul la sfarsit)
        marker_output_dir = os.path.join(args.output_dir, marker)
        os.makedirs(marker_output_dir, exist_ok=True)

        # Folder de lucru pentru antrenare (local sau pe Drive)
        if args.local_workdir:
            marker_workdir = os.path.join(args.local_workdir, marker)
            # Sterge daca exista din rulare precedenta
            import shutil as _shutil
            if os.path.exists(marker_workdir):
                _shutil.rmtree(marker_workdir)
            os.makedirs(marker_workdir, exist_ok=True)
            print(f"Workdir local (rapid): {marker_workdir}")
            print(f"Output final pe Drive: {marker_output_dir}")
        else:
            marker_workdir = marker_output_dir
            print(f"Workdir = output_dir: {marker_workdir}")

        # Incarca model si tokenizer (de fiecare data, pentru a evita poluare intre markeri)
        model, tokenizer = load_and_configure_model(
            args.model_name,
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
            args.target_modules,
            no_lora=args.no_lora,
        )

        # Afiseaza parametri antrenabili
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        # Pregateste dataset-urile
        train_ds = Dataset.from_list(train_data)
        val_ds = Dataset.from_list(val_data)

        # Tokenizare si aliniere etichete
        train_ds = train_ds.map(
            lambda x: align_labels_to_tokens(x, tokenizer, marker, args.max_length),
            batched=True,
            remove_columns=train_ds.column_names,
        )
        val_ds = val_ds.map(
            lambda x: align_labels_to_tokens(x, tokenizer, marker, args.max_length),
            batched=True,
            remove_columns=val_ds.column_names,
        )

        # Data collator pentru padding dinamic
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            padding=True,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=marker_workdir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=args.logging_steps,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=run_entropy,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        # Construieste class weights daca e cazul
        class_weights = None
        if args.class_weight_positive > 1.0:
            # CrossEntropy weight: [weight_label_0, weight_label_1]
            class_weights = torch.tensor([1.0, args.class_weight_positive], dtype=torch.float32)
            print(f"Using WeightedTrainer with class_weights={class_weights.tolist()}")

        if class_weights is not None:
            trainer = WeightedTrainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                processing_class=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
                class_weights=class_weights,
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                processing_class=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
            )

        # Antrenare
        train_result = trainer.train()

        # Evaluare finala
        eval_metrics = trainer.evaluate()
        print(f"\nFinal eval metrics for {marker}:")
        for k, v in eval_metrics.items():
            print(f"  {k}: {v}")

        # Salveaza modelul final mai intai local (sau direct pe Drive daca nu folosim workdir)
        if args.local_workdir:
            # Salveaza in workdir local (rapid)
            local_final_model = os.path.join(marker_workdir, "final_model")
            trainer.save_model(local_final_model)
            tokenizer.save_pretrained(local_final_model)
            print(f"\nSalvat local: {local_final_model}")

            # Copiaza doar final_model + trainer_state.json pe Drive
            import shutil as _shutil
            drive_final_model = os.path.join(marker_output_dir, "final_model")
            if os.path.exists(drive_final_model):
                _shutil.rmtree(drive_final_model)
            print(f"Copiez pe Drive...")
            _shutil.copytree(local_final_model, drive_final_model)

            # Copiaza si trainer_state.json din ultimul checkpoint (pentru analiza ulterioara)
            checkpoints = [d for d in os.listdir(marker_workdir)
                          if d.startswith("checkpoint-")]
            if checkpoints:
                # Ultimul checkpoint dupa numar
                last_ckpt = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
                src_state = os.path.join(marker_workdir, last_ckpt, "trainer_state.json")
                if os.path.exists(src_state):
                    dst_state = os.path.join(marker_output_dir, "trainer_state.json")
                    _shutil.copy2(src_state, dst_state)
                    print(f"Copiat trainer_state.json pe Drive")
            print(f"Salvat pe Drive: {drive_final_model}")

            # Sterge workdir local pentru a elibera spatiu inainte de urmatorul marker
            _shutil.rmtree(marker_workdir)
            print(f"Sters workdir local")
        else:
            # Comportament original (salvare direct pe Drive)
            final_model_dir = os.path.join(marker_output_dir, "final_model")
            trainer.save_model(final_model_dir)
            tokenizer.save_pretrained(final_model_dir)
            print(f"\nSaved final model to {final_model_dir}")

        all_metrics[marker] = eval_metrics

        # Eliberare memorie pentru urmatorul marker
        del model
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Salveaza metricile finale
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
