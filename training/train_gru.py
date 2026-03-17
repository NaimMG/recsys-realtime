import os
import io
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from azure.storage.blob import BlobServiceClient

# ── Config ───────────────────────────────────────────────────────────────
CONN_STR = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
SEQ_LEN = 10        # longueur max de la séquence en entrée
HIDDEN_DIM = 128    # dimension cachée du GRU
BATCH_SIZE = 512
EPOCHS = 5
LR = 0.001
MIN_SEQ_LEN = 3     # séquences trop courtes ignorées

# ── 1. Charger les données et le modèle ALS ──────────────────────────────
print("📥 Chargement des données...")
client = BlobServiceClient.from_connection_string(CONN_STR)

blob = client.get_blob_client("datasets", "retail-rocket/events.csv")
data = blob.download_blob().readall()
df = pd.read_csv(io.BytesIO(data))
df = df[df["event"].isin(["view", "addtocart", "transaction"])]
df = df.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)
print(f"✅ {len(df):,} events chargés")

blob = client.get_blob_client("models", "als_model.pkl")
data = blob.download_blob().readall()
artifacts = pickle.loads(data)
model_als = artifacts["model"]
item_to_idx = artifacts["item_to_idx"]
item_ids = artifacts["item_ids"]
item_embeddings = torch.tensor(model_als.item_factors, dtype=torch.float32)
EMBED_DIM = item_embeddings.shape[1]
print(f"✅ Modèle ALS chargé - embeddings dim={EMBED_DIM}")

# ── 2. Construire les séquences par utilisateur ──────────────────────────
print("\n🔧 Construction des séquences...")
sequences = []
for visitor_id, group in df.groupby("visitorid"):
    items = [i for i in group["itemid"].tolist() if i in item_to_idx]
    if len(items) < MIN_SEQ_LEN:
        continue
    for i in range(MIN_SEQ_LEN - 1, len(items)):
        seq = items[max(0, i - SEQ_LEN):i]
        target = items[i]
        sequences.append((seq, target))

print(f"✅ {len(sequences):,} séquences construites")

# ── 3. Dataset PyTorch ───────────────────────────────────────────────────
class SessionDataset(Dataset):
    def __init__(self, sequences, item_to_idx, item_embeddings, seq_len):
        self.sequences = sequences
        self.item_to_idx = item_to_idx
        self.item_embeddings = item_embeddings
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        # Padding à gauche si la séquence est trop courte
        padded = [0] * (self.seq_len - len(seq)) + [self.item_to_idx[i] for i in seq]
        seq_tensor = self.item_embeddings[padded]
        target_idx = self.item_to_idx[target]
        return seq_tensor, target_idx

dataset = SessionDataset(sequences, item_to_idx, item_embeddings, SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print(f"✅ Dataset prêt - {len(dataset):,} exemples")

# ── 4. Modèle GRU ────────────────────────────────────────────────────────
class SessionGRU(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_items):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.output = nn.Linear(hidden_dim, embed_dim)
        self.n_items = n_items

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]  # dernier état caché
        return self.output(last)  # projeter dans l'espace des embeddings

n_items = len(item_to_idx)
gru_model = SessionGRU(EMBED_DIM, HIDDEN_DIM, n_items)
optimizer = torch.optim.Adam(gru_model.parameters(), lr=LR)
item_emb_matrix = item_embeddings  # pour le calcul de similarité

print(f"\n🚀 Entraînement GRU - {EPOCHS} epochs...")

# ── 5. Entraînement ──────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    gru_model.train()
    total_loss = 0
    n_batches = 0

    for seq_batch, target_batch in loader:
        optimizer.zero_grad()

        # Prédiction du vecteur de session
        pred_vec = gru_model(seq_batch)

        # Loss : similarité cosine avec le vrai item cible
        target_vecs = item_emb_matrix[target_batch]
        loss = 1 - nn.functional.cosine_similarity(pred_vec, target_vecs).mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# ── 6. Export du modèle GRU ──────────────────────────────────────────────
print("\n💾 Export du modèle GRU...")
gru_artifacts = {
    "gru_model_state": gru_model.state_dict(),
    "gru_config": {
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "n_items": n_items,
        "seq_len": SEQ_LEN
    }
}

buffer = io.BytesIO()
pickle.dump(gru_artifacts, buffer)
buffer.seek(0)

blob_client = client.get_blob_client("models", "gru_model.pkl")
blob_client.upload_blob(buffer, overwrite=True)
print("✅ Modèle GRU uploadé sur Blob Storage : models/gru_model.pkl")