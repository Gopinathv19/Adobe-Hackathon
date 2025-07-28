import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

MODEL_DIR    = "tiny_bert"
DATA_PATH    = "headers.csv"
MODEL_NAME   = "prajjwal1/bert-tiny"
MAX_LEN      = 64
BATCH_SIZE   = 32
NUM_EPOCHS   = 5
LR           = 3e-5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HeaderDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        toks = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        feats = torch.tensor([
            row['is_bold'], row['font_size'], row['font_size_rank'],
            row['spacing_before'], row['spacing_after'], row['is_valid_length']
        ], dtype=torch.float)
        label = torch.tensor(row['label'], dtype=torch.long)
        return {
            'input_ids': toks.input_ids.squeeze(0),
            'attention_mask': toks.attention_mask.squeeze(0),
            'extra_feats': feats,
            'labels': label
        }
    
class ImprovedHeaderClassifier(nn.Module):
    def __init__(self, pretrained=MODEL_NAME, n_feats=6, n_classes=4, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained)
        hidden = self.bert.config.hidden_size
        self.feat_proj = nn.Sequential(
            nn.Linear(n_feats, hidden//2), nn.ReLU(), nn.BatchNorm1d(hidden//2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden + hidden//2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, input_ids, attention_mask, extra_feats):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:,0]
        feat_emb = self.feat_proj(extra_feats)
        x = torch.cat([cls_emb, feat_emb], dim=1)
        return self.classifier(x)

# 5. Load & Inference Snippet
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = ImprovedHeaderClassifier().to(DEVICE)
state = torch.load(os.path.join(MODEL_DIR,'best_model.pt'), map_location=DEVICE)
model.load_state_dict(state); model.eval()

import numpy as np
# Read dataset4.csv and extract required columns
def start(CSV_PATH):
    df = pd.read_csv(CSV_PATH)

    # Define required feature columns as per HeaderDataset
    feat_cols = ['is_bold', 'font_size', 'font_size_rank', 'spacing_before', 'spacing_after', 'is_valid_length']

    # Filter out rows with missing text, page_no, or required features
    df = df.dropna(subset=['text', 'page_no'] + feat_cols)

    lines = df['text'].astype(str).tolist()
    feats = df[feat_cols].astype(np.float32).values.tolist()
    pages = df['page_no'].astype(int).tolist()

    # Tokenize
    enc = tokenizer(lines, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
    input_ids = enc.input_ids.to(DEVICE)
    attention_mask = enc.attention_mask.to(DEVICE)
    extra_feats = torch.tensor(feats, dtype=torch.float).to(DEVICE)

    # Predict
    with torch.no_grad():
        logits = model(input_ids, attention_mask, extra_feats)
        preds = logits.argmax(dim=1).cpu().tolist()
    import json
    label_map = {0: "H1", 1: "H2", 2: "H3", 3: "others"}
    results = [
        {"level": label_map.get(int(p), "others"), "text": l, "page": pg}
        for l, p, pg in zip(lines, preds, pages) if int(p) in [0, 1, 2]
    ]
    OUTPUT_JSON = "inference_output.json"
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")
def title(CSV_PATH):
    df = pd.read_csv(CSV_PATH)

    # Define required feature columns as per HeaderDataset
    feat_cols = ['is_bold', 'font_size', 'font_size_rank', 'spacing_before', 'spacing_after', 'is_valid_length']

    # Filter out rows with missing text, page_no, or required features
    df = df.dropna(subset=['text', 'page_no'] + feat_cols)

    lines = df['text'].astype(str).tolist()
    feats = df[feat_cols].astype(np.float32).values.tolist()
    pages = df['page_no'].astype(int).tolist()

    # Tokenize
    enc = tokenizer(lines, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
    input_ids = enc.input_ids.to(DEVICE)
    attention_mask = enc.attention_mask.to(DEVICE)
    extra_feats = torch.tensor(feats, dtype=torch.float).to(DEVICE)

    # Predict
    with torch.no_grad():
        logits = model(input_ids, attention_mask, extra_feats)
        preds = logits.argmax(dim=1).cpu().tolist()
    # Map labels
    label_map = {0: "H1", 1: "H2", 2: "H3", 3: "others"}
    results = [
        {"level": label_map.get(int(p), "others"), "text": l, "page": pg}
        for l, p, pg in zip(lines, preds, pages) if int(p) == 0
    ]
    return results