"""Fine-tune the phoneme recognizer to hear pronunciation errors (v2-3).

Starts from facebook/wav2vec2-lv-60-espeak-cv-ft and keeps its vocabulary and CTC layer. Targets
(scripts/build_finetune_data.py): L2-ARCTIC sentences with what the annotators HEARD (our expected
eSpeak tokens with the annotated errors applied), and CMU ARCTIC native US speech with the expected
tokens, sampled so that non-native and native speech weigh about the same.

After every epoch: PER and mispronunciation-detection F1 on the L2-ARCTIC dev speakers, and PER on a
small native dev set. The best checkpoint by F1 is kept; training stops early when F1 stops
improving or the time budget runs out. Seeded; the loss curve and the dev metrics go to files.

A model fine-tuned on L2-ARCTIC is CC BY-NC 4.0 (non-commercial), like its training data.

Usage (Kaggle, see notebooks/finetune_kaggle.ipynb):
    python scripts/finetune_recognizer.py --data /kaggle/input/<dataset> --out /kaggle/working/recognizer
Smoke test (a few steps on a tiny subset, any machine):
    python scripts/finetune_recognizer.py --data data --manifests data/finetune --out /tmp/ft --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pronunciation.audio import SAMPLE_RATE, to_model_input  # noqa: E402
from pronunciation.metrics import mdd_counts, mdd_metrics, phone_error_rate  # noqa: E402
from pronunciation.phonemes import normalize  # noqa: E402
from pronunciation.recognizer import Decoder  # noqa: E402

BASE_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hf_token() -> str | None:
    """Only needed for private models. On Kaggle it is read from Kaggle Secrets, never from a file."""
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


def read_manifest(folder: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (folder / f"{name}.jsonl").read_text().splitlines()]


class Speech(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], audio_root: Path, tokenizer):
        self.rows, self.audio_root = rows, audio_root
        self.labels = [tokenizer.convert_tokens_to_ids(r["tokens"].split()) for r in rows]
        unknown = tokenizer.unk_token_id
        if any(unknown in ids for ids in self.labels):
            raise ValueError("a target token is not in the recognizer's vocabulary")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        audio, rate = sf.read(self.audio_root / self.rows[i]["audio"], dtype="float32")
        return {"audio": to_model_input(audio, rate), "labels": self.labels[i], "index": i}


class Collate:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch: list[dict]) -> dict:
        inputs = self.feature_extractor([b["audio"] for b in batch], sampling_rate=SAMPLE_RATE, padding=True,
                                        return_attention_mask=True, return_tensors="pt")
        longest = max(len(b["labels"]) for b in batch)
        labels = torch.full((len(batch), longest), -100, dtype=torch.long)
        for k, b in enumerate(batch):
            labels[k, :len(b["labels"])] = torch.tensor(b["labels"])
        return {"input_values": inputs.input_values, "attention_mask": inputs.attention_mask,
                "labels": labels, "index": torch.tensor([b["index"] for b in batch])}


def balanced_sampler(rows: list[dict], seed: int) -> torch.utils.data.WeightedRandomSampler:
    """Non-native and native sentences weigh the same; one epoch = twice the non-native sentences."""
    native = [r["l1"] == "English" for r in rows]
    n_native, n_l2 = sum(native), len(rows) - sum(native)
    weights = [(0.5 / n_native if is_native else 0.5 / n_l2) for is_native in native]
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(weights, num_samples=2 * n_l2, replacement=True, generator=generator)


@torch.no_grad()
def recognize(model, loader, decoder: Decoder, device: str, fp16: bool) -> dict[int, list[str]]:
    """Greedy decoding like the app (restricted to English and learner phonemes), normalized."""
    model.eval()
    out = {}
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            logits = model(batch["input_values"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        frames = model._get_feat_extract_output_lengths(batch["attention_mask"].sum(-1)).tolist()
        for k, i in enumerate(batch["index"].tolist()):
            out[i] = normalize(decoder(log_probs[k, :frames[k]], seconds=0.0).phones)
    model.train()
    return out


def evaluate(model, loaders: dict, rows: dict, decoder: Decoder, device: str, fp16: bool) -> dict:
    l2 = recognize(model, loaders["l2_dev"], decoder, device, fp16)
    counts = sum((mdd_counts(r["expected"], r["perceived"], l2[i], r["function"]) for i, r in enumerate(rows["l2_dev"])),
                 start=Counter())
    mdd = mdd_metrics(counts)
    native = recognize(model, loaders["native_dev"], decoder, device, fp16)
    return {
        "l2_dev_per": phone_error_rate((r["perceived"], l2[i]) for i, r in enumerate(rows["l2_dev"])),
        "l2_dev_f1": mdd["f1"], "l2_dev_precision": mdd["precision"], "l2_dev_recall": mdd["recall"],
        "native_dev_per": phone_error_rate((r["expected"], native[i]) for i, r in enumerate(rows["native_dev"])),
    }


def save(model, feature_extractor, tokenizer, folder: Path, info: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder, safe_serialization=True)
    feature_extractor.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    (folder / "training_info.json").write_text(json.dumps(info, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="root the manifests' audio paths are relative to")
    parser.add_argument("--manifests", type=Path, help="folder with the *.jsonl manifests (default: <data>/manifests)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=15.0, help="leave out longer recordings")
    parser.add_argument("--patience", type=int, default=3, help="epochs without a better dev F1 before stopping")
    parser.add_argument("--max-hours", type=float, default=2.75, help="time budget (Kaggle: stay under 3 h on a T4)")
    parser.add_argument("--seed", type=int, default=762)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="tiny subset, 2 steps, 1 epoch: checks the code end to end")
    args = parser.parse_args()

    from transformers import AutoFeatureExtractor, AutoTokenizer, Wav2Vec2ForCTC, get_linear_schedule_with_warmup

    start = time.time()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda"
    manifests = args.manifests or args.data / "manifests"
    rows = {name: [r for r in read_manifest(manifests, name) if r["seconds"] <= args.max_seconds]
            for name in ("l2arctic_train", "native_train", "l2arctic_dev", "native_dev")}
    train_rows = rows["l2arctic_train"] + rows["native_train"]
    dev_rows = {"l2_dev": rows["l2arctic_dev"], "native_dev": rows["native_dev"]}
    if args.smoke:
        train_rows = rows["l2arctic_train"][:6] + rows["native_train"][:6]
        dev_rows = {"l2_dev": rows["l2arctic_dev"][:4], "native_dev": rows["native_dev"][:2]}
        args.epochs, args.batch_size, args.grad_accum, args.workers = 1, 2, 1, 0

    token = hf_token()
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.base, token=token)
    tokenizer = AutoTokenizer.from_pretrained(args.base, do_phonemize=False, token=token)
    model = Wav2Vec2ForCTC.from_pretrained(args.base, ctc_zero_infinity=True, ctc_loss_reduction="mean", token=token)
    model.freeze_feature_encoder()           # the CNN that turns raw audio into frames stays as it is
    model.gradient_checkpointing_enable()    # recompute activations in the backward pass: less GPU memory
    model.to(device).train()
    vocab = tokenizer.get_vocab()
    decoder = Decoder(vocab, tokenizer.pad_token_id, {i for i in tokenizer.all_special_ids if i != tokenizer.pad_token_id})

    collate = Collate(feature_extractor)
    train_set = Speech(train_rows, args.data, tokenizer)
    sampler = balanced_sampler(train_rows, args.seed)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                                               collate_fn=collate, num_workers=args.workers)
    loaders = {name: torch.utils.data.DataLoader(Speech(r, args.data, tokenizer), batch_size=args.batch_size,
                                                 collate_fn=collate, num_workers=args.workers)
               for name, r in dev_rows.items()}

    steps_per_epoch = 2 if args.smoke else math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path, loss_path = args.out / "train_log.jsonl", args.out / "loss.csv"
    log_path.write_text("")
    with loss_path.open("w", newline="") as f:
        csv.writer(f).writerow(["step", "epoch", "loss", "lr"])

    def log(entry: dict) -> None:
        entry["elapsed_min"] = round((time.time() - start) / 60, 1)
        print(json.dumps(entry), flush=True)
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    info = {"base": args.base, "args": {k: str(v) for k, v in vars(args).items()},
            "train_sentences": {"l2arctic": len(rows["l2arctic_train"]), "native": len(rows["native_train"])},
            "license": "CC BY-NC 4.0 (fine-tuned on L2-ARCTIC); base model Apache-2.0"}
    base = evaluate(model, loaders, dev_rows, decoder, device, fp16)   # the original model, for reference
    log({"epoch": 0, "step": 0, **base})
    best_f1, best_epoch, step = base["l2_dev_f1"], 0, 0

    epoch_seconds = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        losses = []
        for k, batch in enumerate(train_loader):
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                loss = model(batch["input_values"].to(device), attention_mask=batch["attention_mask"].to(device),
                             labels=batch["labels"].to(device)).loss / args.grad_accum
            scaler.scale(loss).backward()
            losses.append(loss.item() * args.grad_accum)
            if (k + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step += 1
                with loss_path.open("a", newline="") as f:
                    csv.writer(f).writerow([step, epoch, round(float(np.mean(losses[-args.grad_accum:])), 4),
                                            scheduler.get_last_lr()[0]])
                if args.smoke and step >= steps_per_epoch * epoch:
                    break
        metrics = evaluate(model, loaders, dev_rows, decoder, device, fp16)
        log({"epoch": epoch, "step": step, "train_loss": round(float(np.mean(losses)), 4), **metrics})
        if metrics["l2_dev_f1"] > best_f1:
            best_f1, best_epoch = metrics["l2_dev_f1"], epoch
            save(model, feature_extractor, tokenizer, args.out / "best", {**info, "epoch": epoch, "dev": metrics})
        epoch_seconds.append(time.time() - epoch_start)
        if epoch - best_epoch >= args.patience:
            print(f"Stopping early: no better dev F1 for {args.patience} epochs.")
            break
        if time.time() - start + max(epoch_seconds) > args.max_hours * 3600:
            print("Stopping: another epoch would exceed the time budget.")
            break
    if best_epoch == 0:
        print(f"No epoch beat the original model (dev F1 {best_f1:.3f}); nothing saved.")
    else:
        print(f"Best dev F1 {best_f1:.3f} at epoch {best_epoch} (original model {base['l2_dev_f1']:.3f}); "
              f"model in {args.out / 'best'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
