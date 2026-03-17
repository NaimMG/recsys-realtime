import os
import io
import pickle
import time
import redis
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
from azure.storage.blob import BlobServiceClient

# ── Config ───────────────────────────────────────────────────────────────
STORAGE_CONN_STR = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
TOP_N = int(os.environ.get("TOP_N_RECOMMENDATIONS", "10"))

# ── Métriques Prometheus ─────────────────────────────────────────────────
REQUEST_COUNT = Counter("recsys_requests_total", "Total recommend requests")
REQUEST_LATENCY = Histogram("recsys_latency_seconds", "Request latency")
CACHE_HITS = Counter("recsys_cache_hits_total", "Redis cache hits")
CACHE_MISSES = Counter("recsys_cache_misses_total", "Redis cache misses")

# ── App FastAPI ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Real-Time Recommendation API",
    description="Moteur de recommandation temps réel basé sur ALS + session vectors",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chargement du modèle au démarrage ────────────────────────────────────
artifacts = None
r = None

@app.on_event("startup")
async def startup():
    global artifacts, r

    print("📥 Chargement du modèle ALS...")
    blob_client = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)
    blob = blob_client.get_blob_client("models", "als_model.pkl")
    data = blob.download_blob().readall()
    artifacts = pickle.loads(data)
    print(f"✅ Modèle chargé - {len(artifacts['item_ids']):,} items")

    print(f"📡 Connexion Redis {REDIS_HOST}:{REDIS_PORT}...")
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    r.ping()
    print("✅ Redis connecté")

# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = TOP_N):
    start = time.time()
    REQUEST_COUNT.inc()

    model = artifacts["model"]
    item_ids = artifacts["item_ids"]
    item_embeddings = model.item_factors

    # Récupérer embedding long terme
    user_vec = None
    user_data = r.get(f"user:{user_id}")
    if user_data:
        CACHE_HITS.inc()
        user_vec = np.frombuffer(user_data, dtype=np.float32)
    else:
        CACHE_MISSES.inc()

    # Récupérer vecteur de session
    session_data = r.get(f"session:{user_id}")
    session_vec = None
    if session_data:
        session_vec = np.frombuffer(session_data, dtype=np.float32)

    # Combiner les deux vecteurs
    if user_vec is not None and session_vec is not None:
        query_vec = (user_vec * 0.7 + session_vec * 0.3)
    elif user_vec is not None:
        query_vec = user_vec
    elif session_vec is not None:
        query_vec = session_vec
    else:
        # Cold start - retourner les items les plus populaires
        popular = artifacts["item_ids"][:n].tolist()
        return {
            "user_id": user_id,
            "recommendations": [int(i) for i in popular],
            "strategy": "popularity",
            "latency_ms": round((time.time() - start) * 1000, 2)
        }

    # Recherche de similarité cosine
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-9
    normalized = item_embeddings / norms
    scores = normalized @ query_vec
    top_indices = np.argpartition(scores, -n)[-n:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    recommendations = [int(item_ids[i]) for i in top_indices]
    latency = round((time.time() - start) * 1000, 2)

    REQUEST_LATENCY.observe(latency / 1000)

    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "strategy": "als+session",
        "latency_ms": latency
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")