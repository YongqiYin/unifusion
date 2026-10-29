"""Train the modality router on a multi-label query set.

Input format (JSON Lines, one object per line)::

    {"query": "which stadium has the largest capacity", "modalities": ["table"]}
    {"query": "who directed the film whose budget was 12 million", "modalities": ["text", "table"]}

Usage::

    python -m unifusion.router.train \\
        --train data/router_train.jsonl \\
        --dev data/router_dev.jsonl \\
        --backbone microsoft/deberta-base \\
        --epochs 3 --batch-size 16 --lr 2e-5 --max-length 256 \\
        --out runs/router

The head is trained with a per-label sigmoid (BCE), so thresholds can be tuned
per label; ``0.5`` is the reported operating point.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from ..types import MODALITIES


# --------------------------------------------------------------- data loading
def read_examples(path):
    """Read ``(texts, multi_hot_labels)`` from a JSONL file."""
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("query") or row.get("text")
            if not text:
                raise ValueError(f"missing 'query' field in {path}")
            positives = row.get("modalities") or []
            labels.append([1 if m in positives else 0 for m in MODALITIES])
            texts.append(text)
    if not texts:
        raise ValueError(f"no examples found in {path}")
    return texts, np.asarray(labels, dtype=np.int64)


# -------------------------------------------------------------------- metrics
def micro_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Micro and macro F1 over the M labels, ignoring zero-denominator labels."""
    true_pos = (y_true * y_pred).sum(axis=0).astype(float)
    pred_pos = y_pred.sum(axis=0).astype(float)
    gold_pos = y_true.sum(axis=0).astype(float)

    def f1(tp, pp, gp):
        if pp + gp == 0:
            return None
        precision = tp / pp if pp else 0.0
        recall = tp / gp if gp else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    micro = f1(true_pos.sum(), pred_pos.sum(), gold_pos.sum()) or 0.0
    per_label = [v for v in (f1(tp, pp, gp) for tp, pp, gp in zip(true_pos, pred_pos, gold_pos))
                 if v is not None]
    macro = float(np.mean(per_label)) if per_label else 0.0
    return float(micro), macro


def decode(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold the sigmoid outputs; fall back to arg-max when nothing fires."""
    predictions = (probs >= threshold).astype(np.int64)
    empty = predictions.sum(axis=1) == 0
    if empty.any():
        predictions[empty] = 0
        predictions[empty, np.argmax(probs[empty], axis=1)] = 1
    return predictions


# ------------------------------------------------------------------- training
def train(args) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer

    from .model import ModalityRouter, _build_module

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_texts, train_labels = read_examples(args.train)
    dev_texts, dev_labels = read_examples(args.dev)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    class _Dataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(texts, truncation=True, padding="max_length",
                                       max_length=args.max_length)
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
            return item

    train_loader = DataLoader(_Dataset(train_texts, train_labels),
                              batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(_Dataset(dev_texts, dev_labels),
                            batch_size=args.batch_size, shuffle=False)

    model = _build_module(args.backbone, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                  end_factor=0.1, total_iters=total_steps)

    @torch.no_grad()
    def evaluate():
        model.eval()
        probabilities = []
        for batch in dev_loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        model.train()
        return np.concatenate(probabilities, axis=0)

    history, best_macro, best_state = [], -1.0, None
    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.item())
            if step % max(1, len(train_loader) // 5) == 0:
                print(f"[epoch {epoch}] step {step}/{len(train_loader)} "
                      f"loss={running / step:.4f}", flush=True)

        probabilities = evaluate()
        micro, macro = micro_macro_f1(dev_labels, decode(probabilities, args.threshold))
        history.append({"epoch": epoch, "train_loss": running / len(train_loader),
                        "dev_micro_f1": micro, "dev_macro_f1": macro})
        print(f"[epoch {epoch}] dev micro-F1={micro:.4f} macro-F1={macro:.4f}", flush=True)
        if macro > best_macro:
            best_macro = macro
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    router = ModalityRouter(backbone=args.backbone, dropout=args.dropout,
                            max_length=args.max_length, device=device)
    router._module, router._tokenizer = model, tokenizer
    if best_state is not None:
        model.load_state_dict(best_state)
    router.save(out_dir)

    summary = {"backbone": args.backbone, "epochs": args.epochs,
               "batch_size": args.batch_size, "lr": args.lr,
               "max_length": args.max_length, "threshold": args.threshold,
               "train_rows": len(train_texts), "dev_rows": len(dev_texts),
               "best_dev_macro_f1": best_macro, "history": history}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="training JSONL")
    parser.add_argument("--dev", required=True, help="validation JSONL")
    parser.add_argument("--backbone", default="microsoft/deberta-base")
    parser.add_argument("--out", default="runs/router")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    summary = train(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
