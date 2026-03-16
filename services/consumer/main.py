import os
import io
import json
import pickle
import redis
import numpy as np
from azure.storage.blob import BlobServiceClient
from azure.eventhub import EventHubConsumerClient

# ── Config ───────────────────────────────────────────────────────────────
EVENTHUB_CONN_STR = os.environ["EVENTHUB_CONNECTION_STRING"]
EVENTHUB_NAME = os.environ.get("EVENTHUB_NAME_EVENTS", "user-events")
STORAGE_CONN_STR = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
SESSION_WINDOW = 30  # nombre max d'items par session

# ── 1. Charger le modèle ALS depuis Blob Storage ─────────────────────────
print("📥 Chargement du modèle ALS depuis Blob Storage...")
blob_client = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)
blob = blob_client.get_blob_client("models", "als_model.pkl")
data = blob.download_blob().readall()
artifacts = pickle.loads(data)

model = artifacts["model"]
item_to_idx = artifacts["item_to_idx"]
item_embeddings = model.item_factors
print(f"✅ Modèle chargé - {len(item_to_idx):,} items")

# ── 2. Connexion Redis ────────────────────────────────────────────────────
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
r.ping()
print(f"✅ Redis connecté sur {REDIS_HOST}:{REDIS_PORT}")

# ── 3. Fonctions utilitaires ─────────────────────────────────────────────
def get_session_vector(visitorid: int) -> np.ndarray:
    """Récupère le vecteur de session depuis Redis."""
    key = f"session:{visitorid}"
    data = r.get(key)
    if data:
        return np.frombuffer(data, dtype=np.float32)
    return None

def update_session_vector(visitorid: int, itemid: int):
    """Met à jour le vecteur de session avec le nouvel item."""
    if itemid not in item_to_idx:
        return
    idx = item_to_idx[itemid]
    item_vec = item_embeddings[idx].astype(np.float32)

    current = get_session_vector(visitorid)
    if current is None:
        new_vec = item_vec
    else:
        new_vec = (current + item_vec) / 2.0

    r.setex(
        f"session:{visitorid}",
        3600,  # expire après 1 heure
        new_vec.tobytes()
    )

def store_user_embedding(visitorid: int):
    """Stocke l'embedding long terme de l'utilisateur dans Redis."""
    user_to_idx = artifacts["user_to_idx"]
    if visitorid not in user_to_idx:
        return
    idx = user_to_idx[visitorid]
    user_vec = model.user_factors[idx].astype(np.float32)
    r.setex(
        f"user:{visitorid}",
        86400,  # expire après 24 heures
        user_vec.tobytes()
    )

# ── 4. Traitement des events ─────────────────────────────────────────────
def on_event(partition_context, event):
    try:
        data = json.loads(event.body_as_str())
        visitorid = data["visitorid"]
        itemid = data["itemid"]
        event_type = data["event"]

        update_session_vector(visitorid, itemid)
        store_user_embedding(visitorid)

        partition_context.update_checkpoint(event)
    except Exception as e:
        print(f"❌ Erreur traitement event: {e}")

# ── 5. Démarrer le consumer ──────────────────────────────────────────────
print("🚀 Consumer démarré, en attente d'events...")
consumer = EventHubConsumerClient.from_connection_string(
    conn_str=EVENTHUB_CONN_STR,
    consumer_group="$Default",
    eventhub_name=EVENTHUB_NAME,
)

with consumer:
    consumer.receive(
        on_event=on_event,
        starting_position="-1",  # depuis le début
    )