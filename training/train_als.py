import os
import io
import pickle
import numpy as np
import scipy.sparse as sparse
from azure.storage.blob import BlobServiceClient
from implicit import als
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────
CONN_STR = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
CONTAINER_DATASETS = "datasets"
CONTAINER_MODELS = "models"
MIN_INTERACTIONS = 5  # filtrer les users/items trop rares

# ── 1. Charger events.csv depuis Blob Storage ───────────────────────────
print("📥 Chargement events.csv depuis Blob Storage...")
client = BlobServiceClient.from_connection_string(CONN_STR)
blob = client.get_blob_client(CONTAINER_DATASETS, "retail-rocket/events.csv")
data = blob.download_blob().readall()
df = pd.read_csv(io.BytesIO(data))
print(f"✅ {len(df):,} events chargés")
print(df.head())

# ── 2. Préparer les données ──────────────────────────────────────────────
print("\n🔧 Préparation des données...")

# Garder seulement les events pertinents
df = df[df["event"].isin(["view", "addtocart", "transaction"])]

# Poids par type d'event
weights = {"view": 1, "addtocart": 3, "transaction": 5}
df["weight"] = df["event"].map(weights)

# Filtrer users et items avec trop peu d'interactions
user_counts = df["visitorid"].value_counts()
item_counts = df["itemid"].value_counts()
df = df[
    df["visitorid"].isin(user_counts[user_counts >= MIN_INTERACTIONS].index) &
    df["itemid"].isin(item_counts[item_counts >= MIN_INTERACTIONS].index)
]
print(f"✅ {len(df):,} events après filtrage")
print(f"   {df['visitorid'].nunique():,} utilisateurs")
print(f"   {df['itemid'].nunique():,} items")

# ── 3. Construire la matrice user-item ───────────────────────────────────
print("\n🔧 Construction de la matrice user-item...")

user_ids = df["visitorid"].unique()
item_ids = df["itemid"].unique()
user_to_idx = {u: i for i, u in enumerate(user_ids)}
item_to_idx = {it: i for i, it in enumerate(item_ids)}

rows = df["visitorid"].map(user_to_idx)
cols = df["itemid"].map(item_to_idx)
values = df["weight"].values

matrix = sparse.csr_matrix(
    (values, (rows, cols)),
    shape=(len(user_ids), len(item_ids))
)
print(f"✅ Matrice {matrix.shape[0]:,} x {matrix.shape[1]:,}")

# ── 4. Entraîner le modèle ALS ───────────────────────────────────────────
print("\n🚀 Entraînement ALS...")
model = als.AlternatingLeastSquares(
    factors=64,
    iterations=20,
    regularization=0.1,
    use_gpu=False
)
model.fit(matrix)
print("✅ Modèle entraîné")

# ── 5. Exporter les embeddings ───────────────────────────────────────────
print("\n💾 Export des embeddings...")
artifacts = {
    "model": model,
    "user_ids": user_ids,
    "item_ids": item_ids,
    "user_to_idx": user_to_idx,
    "item_to_idx": item_to_idx,
    "matrix": matrix,
}

buffer = io.BytesIO()
pickle.dump(artifacts, buffer)
buffer.seek(0)

blob_client = client.get_blob_client(CONTAINER_MODELS, "als_model.pkl")
blob_client.upload_blob(buffer, overwrite=True)
print("✅ Modèle uploadé sur Blob Storage : models/als_model.pkl")