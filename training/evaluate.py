import os
import io
import pickle
import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient

# ── Config ───────────────────────────────────────────────────────────────
CONN_STR = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
K = 10  # top-K recommandations
MIN_INTERACTIONS = 5

# ── 1. Charger les données ────────────────────────────────────────────────
print("📥 Chargement des données...")
blob_client = BlobServiceClient.from_connection_string(CONN_STR)
blob = blob_client.get_blob_client("datasets", "retail-rocket/events.csv")
data = blob.download_blob().readall()
df = pd.read_csv(io.BytesIO(data))
df = df[df["event"].isin(["view", "addtocart", "transaction"])]
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"✅ {len(df):,} events chargés")

# ── 2. Split temporel 80/20 ──────────────────────────────────────────────
print("\n✂️ Split temporel 80/20...")
split_idx = int(len(df) * 0.8)
split_ts = df.iloc[split_idx]["timestamp"]
train_df = df[df["timestamp"] < split_ts]
test_df = df[df["timestamp"] >= split_ts]

print(f"   Train : {len(train_df):,} events jusqu'au {pd.to_datetime(split_ts, unit='ms').date()}")
print(f"   Test  : {len(test_df):,} events après le {pd.to_datetime(split_ts, unit='ms').date()}")

# ── 3. Charger le modèle ─────────────────────────────────────────────────
print("\n📥 Chargement du modèle ALS...")
blob = blob_client.get_blob_client("models", "als_model.pkl")
data = blob.download_blob().readall()
artifacts = pickle.loads(data)
model = artifacts["model"]
user_to_idx = artifacts["user_to_idx"]
item_to_idx = artifacts["item_to_idx"]
item_ids = artifacts["item_ids"]
print(f"✅ Modèle chargé")

# ── 4. Construire le ground truth depuis le test set ─────────────────────
print("\n🔧 Construction du ground truth...")
test_user_items = (
    test_df[test_df["visitorid"].isin(user_to_idx)]
    .groupby("visitorid")["itemid"]
    .apply(set)
    .to_dict()
)
print(f"✅ {len(test_user_items):,} utilisateurs dans le test set")

# ── 5. Calcul des métriques ──────────────────────────────────────────────
print(f"\n📊 Calcul des métriques @{K}...")

precisions, recalls, ndcgs = [], [], []
sample_users = list(test_user_items.keys())[:500]  # échantillon de 500 users

for user_id in sample_users:
    if user_id not in user_to_idx:
        continue

    user_idx = user_to_idx[user_id]
    true_items = test_user_items[user_id]

    # Générer les recommandations
    user_vec = model.user_factors[user_idx]
    scores = model.item_factors @ user_vec
    top_k_idx = np.argpartition(scores, -K)[-K:]
    top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
    recommended = [int(item_ids[i]) for i in top_k_idx]

    # Precision@K
    hits = len(set(recommended) & true_items)
    precisions.append(hits / K)

    # Recall@K
    recalls.append(hits / len(true_items) if true_items else 0)

    # NDCG@K
    dcg = sum(
        1 / np.log2(i + 2)
        for i, item in enumerate(recommended)
        if item in true_items
    )
    ideal_hits = min(len(true_items), K)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    ndcgs.append(dcg / idcg if idcg > 0 else 0)

# ── 6. Résultats ─────────────────────────────────────────────────────────
print("\n" + "="*40)
print(f"📈 RÉSULTATS DU MODÈLE ALS @{K}")
print("="*40)
print(f"Precision@{K}  : {np.mean(precisions):.4f}")
print(f"Recall@{K}     : {np.mean(recalls):.4f}")
print(f"NDCG@{K}       : {np.mean(ndcgs):.4f}")
print("="*40)
print(f"\n✅ Évaluation terminée sur {len(precisions)} utilisateurs")