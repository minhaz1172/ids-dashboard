# model.py
# CNN-BiLSTM-Transformer for Network Intrusion Detection
# Exact architecture from NB3b — renamed to HybridModel for Streamlit dashboard
# Parameters: 291,980 | Input: (B, 1, 31) | Output: (B, 12)

import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_ch=1, ch=None, k=3, drop=0.3):
        super().__init__()
        if ch is None:
            ch = [64, 128]
        layers, c = [], in_ch
        for co in ch:
            layers += [
                nn.Conv1d(c, co, k, padding=k // 2),
                nn.BatchNorm1d(co),
                nn.ReLU(inplace=True),
            ]
            c = co
        layers.append(nn.Dropout(drop))
        self.net    = nn.Sequential(*layers)
        self.out_ch = ch[-1]

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_in, dm=64, nh=2, ffn=128, n=1, drop=0.3):
        super().__init__()
        self.proj = nn.Linear(d_in, dm) if d_in != dm else nn.Identity()
        enc = nn.TransformerEncoderLayer(
            d_model=dm, nhead=nh, dim_feedforward=ffn,
            dropout=drop, batch_first=True, norm_first=True,
        )
        self.enc  = nn.TransformerEncoder(enc, n)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.enc(self.proj(x)).mean(dim=1))


class HybridModel(nn.Module):
    """
    CNN + BiLSTM + Transformer for Network Intrusion Detection.
    Exact architecture from NB3b (renamed from CNN_BiLSTM_Transformer).

    Input  : (batch, 1, n_features)   e.g. (1, 1, 31)
    Output : (batch, n_classes)        e.g. (1, 12)

    Architecture:
        CNN  → local feature extraction  (1 → 64 → 128 channels)
        Pool → global average over feature positions
        BiLSTM → sequential context      (128 → 64×2 hidden)
        Transformer → global attention   (128 → 64 d_model)
        Concat(CNN_pool, Transformer) → FC → logits
    """

    def __init__(self, input_dim=31, num_classes=12):
        super().__init__()
        # CNN block
        self.cnn       = CNNBlock(1, [64, 128], k=3, drop=0.3)
        cnn_out        = 128

        # BiLSTM
        self.lstm      = nn.LSTM(
            cnn_out, 64, num_layers=2,
            batch_first=True, bidirectional=True,
            dropout=0.3,
        )
        self.lstm_drop = nn.Dropout(0.3)

        # Transformer
        self.trans = TransformerBlock(
            d_in=128, dm=64, nh=2, ffn=128, n=1, drop=0.3
        )

        # Global pooling (CNN path)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classifier: concat(CNN_pool=128, Transformer=64) → 128 → num_classes
        self.fc = nn.Sequential(
            nn.Linear(cnn_out + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, n_features)
        c   = self.cnn(x)                    # (B, 128, n_features)
        cp  = self.pool(c).squeeze(-1)       # (B, 128)  — CNN global pool
        s,_ = self.lstm(c.permute(0, 2, 1)) # (B, n_features, 128)
        t   = self.trans(self.lstm_drop(s))  # (B, 64)   — Transformer
        return self.fc(torch.cat([cp, t], dim=1))