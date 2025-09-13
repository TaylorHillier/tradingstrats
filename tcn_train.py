# tcn_train.py (improved: scaling, time split, clipping, metrics, saved stats)

import argparse, math, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

FEATURE_COLS = ["OpenT","HighT","LowT","CloseT","Range","Return",
                "SMA_F","SMA_S","SMA_Diff","RSI","ADX","Volume","Delta"]
LABEL_COLS = ["LabelLong","LabelShort"]

class SeqDataset(Dataset):
    def __init__(self, Xseq, Y):
        self.Xseq = Xseq.astype(np.float32)
        self.Y = Y.astype(np.float32)
    def __len__(self): return len(self.Y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.Xseq[idx]), torch.from_numpy(self.Y[idx])

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size=2, num_channels=(64, 64, 64), kernel_size=3, dropout=0.2):
        super().__init__()
        layers, in_ch = [], input_size
        for i, out_ch in enumerate(num_channels):
            layers += [TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=2**i, dropout=dropout)]
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, output_size)
    def forward(self, x):           # x: [B, T, F]
        x = x.transpose(1, 2)       # -> [B, F, T]
        h = self.network(x)         # -> [B, C, T]
        h_last = h[:, :, -1]        # last step
        return self.fc(h_last)      # [B, 2]

def make_sequences(df, seq_len):
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COLS].values.astype(np.float32)
    Xseq, Y = [], []
    for i in range(len(df) - seq_len):
        Xseq.append(X[i:i+seq_len])
        Y.append(y[i+seq_len])
    return np.stack(Xseq), np.stack(Y)

def time_split(Xseq, Y, train_ratio=0.8):
    n = len(Y)
    n_train = int(n * train_ratio)
    return (Xseq[:n_train], Y[:n_train]), (Xseq[n_train:], Y[n_train:])

def standardize_fit(X_train):          # X_train: [N, T, F]
    # compute mean/std per feature across all windows+timesteps
    mu = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    sd = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)  # avoid div-by-zero
    return mu.astype(np.float32), sd.astype(np.float32)

def standardize_apply(X, mu, sd):
    return (X - mu) / sd

def accuracy_from_logits(logits, one_hot_targets):
    pred = logits.argmax(dim=1)
    targ = one_hot_targets.argmax(dim=1)
    return (pred == targ).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=r"C:\tradingstrats\training_data.csv")
    ap.add_argument("--seq_len", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--model_out", default="tcn_model.pt")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # (Optional) sanity-check labels:
    lbl_long_rate = df["LabelLong"].mean()
    print(f"LabelLong mean (class balance): {lbl_long_rate:.3f}")

    Xseq, Y = make_sequences(df, args.seq_len)

    # chronological split (no leakage)
    (Xtr, Ytr), (Xv, Yv) = time_split(Xseq, Y, 0.8)

    # standardize using TRAIN stats only
    mu, sd = standardize_fit(Xtr)
    Xtr = standardize_apply(Xtr, mu, sd)
    Xv  = standardize_apply(Xv,  mu, sd)

    train_ds = SeqDataset(Xtr, Ytr)
    val_ds   = SeqDataset(Xv,  Yv)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCN(input_size=len(FEATURE_COLS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    best_val = math.inf
    patience, bad = 8, 0

    for ep in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        tr_loss_sum, tr_correct, tr_n = 0.0, 0, 0
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            targ = Yb.argmax(dim=1)
            logits = model(Xb)
            loss = crit(logits, targ)
            opt.zero_grad()
            loss.backward()
            if args.clip > 0: nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tr_loss_sum += float(loss.item()) * Xb.size(0)
            tr_correct += (logits.argmax(dim=1) == targ).sum().item()
            tr_n += Xb.size(0)
        tr_loss = tr_loss_sum / max(tr_n,1)
        tr_acc  = tr_correct / max(tr_n,1)

        # ---- val ----
        model.eval()
        va_loss_sum, va_correct, va_n = 0.0, 0, 0
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                targ = Yb.argmax(dim=1)
                logits = model(Xb)
                loss = crit(logits, targ)
                va_loss_sum += float(loss.item()) * Xb.size(0)
                va_correct += (logits.argmax(dim=1) == targ).sum().item()
                va_n += Xb.size(0)
        va_loss = va_loss_sum / max(va_n,1)
        va_acc  = va_correct / max(va_n,1)

        print(f"Epoch {ep}/{args.epochs}  "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"val_loss={va_loss:.4f} acc={va_acc:.3f}")

        # early stopping
        if va_loss < best_val - 1e-4:
            best_val, bad = va_loss, 0
            torch.save({
                "model_state": model.state_dict(),
                "feature_cols": FEATURE_COLS,
                "seq_len": args.seq_len,
                "mu": mu, "sd": sd
            }, args.model_out)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print(f"Saved best model to {args.model_out} (best val_loss={best_val:.4f})")

if __name__ == "__main__":
    main()

# python tcn_train.py --csv "C:\tradingstrats\training_data.csv" --seq_len 50 --epochs 10
