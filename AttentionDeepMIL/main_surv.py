"""Bước 7: train ABMIL + Cox loss, 5-fold CV theo site, đánh giá bằng C-index.

Ví dụ:
    python AttentionDeepMIL/main_surv.py \
        --labels data/TCGA-LUAD/labels/os.csv \
        --feat_dir data/TCGA-LUAD/features/pt_files \
        --split_dir data/TCGA-LUAD/splits \
        --out_dir results/TCGA-LUAD_abmil
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from losses import c_index, cox_ph_loss
from model import ABMILSurv
from survival_dataset import SurvivalBags, collate_bags


def parse_args():
    p = argparse.ArgumentParser(description="ABMIL survival (Cox)")
    p.add_argument("--labels", required=True)
    p.add_argument("--feat_dir", required=True)
    p.add_argument("--split_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--folds", type=int, nargs="*", default=None, help="mặc định: mọi fold trong split_dir")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--reg", type=float, default=1e-5, help="weight decay")
    p.add_argument("--bag_batch", type=int, default=32, help="số bệnh nhân / bước tối ưu (cho Cox loss)")
    p.add_argument("--max_patches", type=int, default=4096, help="lấy mẫu patch khi train; 0 = dùng hết")
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--no_gated", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def run_epoch(model, loader, device, optimizer=None):
    """Train (khi có optimizer) hoặc eval một epoch. Trả về loss TB, C-index, và dự đoán."""
    train = optimizer is not None
    model.train(train)
    losses, risks, times, events, cases, attns = [], [], [], [], [], {}
    for bags, t, e, ids in loader:
        with torch.set_grad_enabled(train):
            out = [model(b.to(device)) for b in bags]
            risk = torch.stack([r for r, _ in out])
            loss = cox_ph_loss(risk, t, e)
            if train and loss is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if loss is not None:
            losses.append(loss.item())
        risks += risk.detach().cpu().tolist()
        times += t.tolist()
        events += e.tolist()
        cases += ids
        if not train:
            attns.update({cid: a.squeeze(0).cpu().numpy() for cid, (_, a) in zip(ids, out)})
    ci = c_index(risks, times, events)
    preds = pd.DataFrame({"case_id": cases, "risk": risks, "time": times, "event": events})
    return float(np.mean(losses)) if losses else float("nan"), ci, preds, attns


def train_fold(k, args, device):
    split = pd.read_csv(os.path.join(args.split_dir, f"fold_{k}.csv"))
    tr_ids = split.loc[split["split"] == "train", "case_id"]
    va_ids = split.loc[split["split"] == "val", "case_id"]
    tr_ds = SurvivalBags(args.labels, args.feat_dir, tr_ids, max_patches=args.max_patches or None, seed=args.seed)
    va_ds = SurvivalBags(args.labels, args.feat_dir, va_ids)
    print(f"\n=== Fold {k}: train {len(tr_ds)} ({tr_ds.events.sum()} biến cố), "
          f"val {len(va_ds)} ({va_ds.events.sum()} biến cố) ===")

    kw = dict(collate_fn=collate_bags, num_workers=args.num_workers)
    tr_loader = DataLoader(tr_ds, batch_size=args.bag_batch, shuffle=True,
                           drop_last=len(tr_ds) > args.bag_batch, **kw)
    va_loader = DataLoader(va_ds, batch_size=args.bag_batch, shuffle=False, **kw)

    input_dim = tr_ds[0][0].shape[1]
    model = ABMILSurv(input_dim=input_dim, gated=not args.no_gated, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)

    fold_dir = os.path.join(args.out_dir, f"fold_{k}")
    os.makedirs(fold_dir, exist_ok=True)
    best_ci, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ci, _, _ = run_epoch(model, tr_loader, device, optimizer)
        va_loss, va_ci, preds, attns = run_epoch(model, va_loader, device)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_ci=tr_ci, val_loss=va_loss, val_ci=va_ci))
        flag = ""
        if va_ci > best_ci:
            best_ci, flag = va_ci, " *"
            torch.save(model.state_dict(), os.path.join(fold_dir, "best.pt"))
            preds.to_csv(os.path.join(fold_dir, "val_preds.csv"), index=False)
            np.savez_compressed(os.path.join(fold_dir, "val_attention.npz"), **attns)
        print(f"ep {epoch:3d} | train loss {tr_loss:.4f} ci {tr_ci:.4f} | "
              f"val loss {va_loss:.4f} ci {va_ci:.4f}{flag}")

    # C-index của epoch cuối không phụ thuộc chọn checkpoint trên val -> ước lượng ít lạc quan hơn
    torch.save(model.state_dict(), os.path.join(fold_dir, "last.pt"))
    pd.DataFrame(history).to_csv(os.path.join(fold_dir, "history.csv"), index=False)
    return best_ci, history[-1]["val_ci"]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    folds = args.folds
    if folds is None:
        folds = sorted(int(os.path.basename(p)[5:-4]) for p in glob.glob(os.path.join(args.split_dir, "fold_*.csv")))
    print(f"device: {device}, folds: {folds}")

    rows = []
    for k in folds:
        best, last = train_fold(k, args, device)
        rows.append(dict(fold=k, best_val_ci=best, last_val_ci=last))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    print("\n" + res.to_string(index=False))
    print(f"C-index (epoch cuối): {res.last_val_ci.mean():.4f} ± {res.last_val_ci.std():.4f}")
    print(f"C-index (best epoch, lạc quan): {res.best_val_ci.mean():.4f} ± {res.best_val_ci.std():.4f}")


if __name__ == "__main__":
    main()
