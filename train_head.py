"""Classification head training for MORTM E3 task.
Trains a linear classifier on top of the mean-pooled hidden states of the frozen/unfrozen backbone.
"""
import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from flash_attn.bert_padding import pad_input, unpad_input

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_MUSIC
from mortm.models.mortm import MORTM
from mortm.models.modules.config import MORTMArgs
from mortm.models.modules.progress import LearningProgress

class _DefaultLearningProgress(LearningProgress):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def get_device(self):
        return self.device

def build_class_sets(tok):
    keys = sorted(i for n, i in tok.tokens.items() if str(n).startswith("k_"))
    dens = sorted(i for n, i in tok.tokens.items()
                  if ("DENSE" in str(n).upper() or "DENSITY" in str(n).upper()
                      or str(n).upper().startswith("<NOTE_DENSE")))
    genres = sorted(i for n, i in tok.tokens.items() if str(n).startswith("<GENRE_"))
    return np.array(keys), np.array(dens), np.array(genres)

class AnalysisDataset(Dataset):
    def __init__(self, json_path, tok, task_name):
        self.samples = []
        
        # Build token to class index mapping
        key_tokens, dense_tokens, genre_tokens = build_class_sets(tok)
        self.key_map = {tok_id: idx for idx, tok_id in enumerate(key_tokens)}
        self.dense_map = {tok_id: idx for idx, tok_id in enumerate(dense_tokens)}
        self.genre_map = {tok_id: idx for idx, tok_id in enumerate(genre_tokens)}

        self.key_trig = tok.get("<KEY>")
        self.dense_trig = tok.get("<DENCE>")
        self.genre_trig = tok.get("<GENRE>")
        self.meta_trig = tok.get("<META>")

        with open(json_path) as f:
            paths = json.load(f)

        print(f"Loading {len(paths)} files for task '{task_name}'...")
        for p in paths:
            if not os.path.exists(p):
                continue
            try:
                with np.load(p, allow_pickle=True) as data:
                    i = 1
                    while True:
                        arr_key = f"array{i}"
                        if arr_key not in data.files:
                            break
                        seq = data[arr_key]
                        if seq.ndim == 0:
                            i += 1
                            continue
                        
                        # Process sequence
                        seq_list = seq.tolist()
                        
                        if task_name == "key":
                            if self.key_trig in seq_list:
                                idx = seq_list.index(self.key_trig)
                                if idx + 1 < len(seq_list):
                                    tgt_tok = seq_list[idx + 1]
                                    if tgt_tok in self.key_map:
                                        music_block = seq_list[1:idx] # skip leading <EOS>
                                        self.samples.append((music_block, self.key_map[tgt_tok]))
                            elif self.meta_trig in seq_list:
                                idx = seq_list.index(self.meta_trig)
                                music_block = seq_list[1:idx]
                                # Look for key token in system block
                                for tok_id in seq_list[idx + 1:]:
                                    if tok_id in self.key_map:
                                        self.samples.append((music_block, self.key_map[tok_id]))
                                        break
                        elif task_name == "dense":
                            if self.dense_trig in seq_list:
                                idx = seq_list.index(self.dense_trig)
                                if idx + 2 < len(seq_list):
                                    tgt_tok = seq_list[idx + 2] # skip <INST_PIANO> or <INST_SAX>
                                    if tgt_tok in self.dense_map:
                                        music_block = seq_list[1:idx]
                                        self.samples.append((music_block, self.dense_map[tgt_tok]))
                            elif self.meta_trig in seq_list:
                                idx = seq_list.index(self.meta_trig)
                                music_block = seq_list[1:idx]
                                # Look for density token in system block
                                for tok_id in seq_list[idx + 1:]:
                                    if tok_id in self.dense_map:
                                        self.samples.append((music_block, self.dense_map[tok_id]))
                                        break
                        elif task_name == "genre":
                            if self.genre_trig in seq_list:
                                idx = seq_list.index(self.genre_trig)
                                if idx + 1 < len(seq_list):
                                    tgt_tok = seq_list[idx + 1]
                                    if tgt_tok in self.genre_map:
                                        music_block = seq_list[1:idx]
                                        self.samples.append((music_block, self.genre_map[tgt_tok]))
                            elif self.meta_trig in seq_list:
                                idx = seq_list.index(self.meta_trig)
                                music_block = seq_list[1:idx]
                                # Look for genre token in system block
                                for tok_id in seq_list[idx + 1:]:
                                    if tok_id in self.genre_map:
                                        self.samples.append((music_block, self.genre_map[tok_id]))
                                        break
                        
                        i += 1
            except Exception as e:
                # print(f"Error loading {p}: {e}")
                pass
                
        print(f"Successfully loaded {len(self.samples)} samples for task '{task_name}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    padded_xs = []
    for x in xs:
        pad_len = max_len - len(x)
        padded_xs.append(F.pad(x, (0, pad_len), value=0))
    x_batch = torch.stack(padded_xs)
    y_batch = torch.stack(ys)
    mask = (x_batch != 0)
    return x_batch, y_batch, mask

class ClassifierModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.d_model, num_classes)

    def forward(self, x, padding_mask):
        # Embed and unpad
        x_embed = self.backbone.embedding(x).to(dtype=torch.bfloat16)
        batch, tgt_len, embed_dim = x_embed.size()
        x_unpad, indices, cu_seqlens, max_s, used_seqlens = unpad_input(x_embed, padding_mask)
        
        # Run through decoder
        out = self.backbone.decoder(
            tgt=x_unpad,
            tgt_is_causal=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_s,
            batch_size=batch,
            indices=indices,
            is_save_cache=False
        )
        
        # Pad back to original sequence length
        out = pad_input(out, indices, batch, tgt_len) # [B, S, d_model]
        
        # Mean pooling excluding padding tokens
        mask = padding_mask.unsqueeze(-1).float() # [B, S, 1]
        pooled = (out.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1) # [B, d_model]
        
        # Classification head
        logits = self.head(pooled)
        return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, required=True, choices=["A1","A2","A1-40B-80M","A1-40B-160M"], help="Model arm")
    parser.add_argument("--task", type=str, required=True, choices=["key", "dense", "genre"], help="Task name")
    parser.add_argument("--backbone_cfg", type=str, default="configs/models/mortm/foundation/80M.json")
    parser.add_argument("--backbone_ckpt", type=str, required=True, help="Path to backbone .pth file")
    parser.add_argument("--train_json", type=str, default="/home/takaaki-nagoshi/data/sft/analysis/train.json")
    parser.add_argument("--eval_json", type=str, default="/home/takaaki-nagoshi/data/sft/analysis/eval.json")
    parser.add_argument("--save_dir", type=str, default="out/models/paper/E3")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42, help="乱数シード(複数シード頑健性評価用)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  seed={args.seed}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize tokenizer and get number of classes
    tok = Tokenizer(get_token_converter_pro(TO_MUSIC))
    tok.mode(TO_MUSIC)
    key_tokens, dense_tokens, genre_tokens = build_class_sets(tok)
    
    num_classes_dict = {
        "key": len(key_tokens),
        "dense": len(dense_tokens),
        "genre": len(genre_tokens)
    }
    num_classes = num_classes_dict[args.task]
    print(f"Training on task '{args.task}' with {num_classes} classes")

    # Load datasets
    train_dataset = AnalysisDataset(args.train_json, tok, args.task)
    eval_dataset = AnalysisDataset(args.eval_json, tok, args.task)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize backbone
    prog = _DefaultLearningProgress()
    backbone_args = MORTMArgs(args.backbone_cfg)
    backbone = MORTM(backbone_args, prog)
    
    print(f"Loading backbone from {args.backbone_ckpt}")
    backbone.load_state_dict(torch.load(args.backbone_ckpt, map_location=device))

    # Initialize full model
    model = ClassifierModel(backbone, num_classes).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    _suffix = f"_s{args.seed}" if args.seed != 42 else ""
    task_save_dir = os.path.join(args.save_dir, f"{args.arm}_{args.task}{_suffix}")
    os.makedirs(task_save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for step, (x, y, mask) in enumerate(train_loader):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x, mask)
                loss = criterion(logits, y)
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total if total > 0 else 0.0
        train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # Eval
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for x, y, mask in eval_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x, mask)
                    loss = criterion(logits, y)
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_loss /= len(eval_loader) if len(eval_loader) > 0 else 1.0

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

        # Save checkpoint if best
        if val_acc >= best_acc:
            best_acc = val_acc
            save_path = os.path.join(task_save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Val Acc {best_acc:.4f} to {save_path}")

    print(f"Training completed. Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()
