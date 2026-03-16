import os
import io
import json
import time
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.eventhub import EventHubProducerClient, EventData

# ── Config ───────────────────────────────────────────────────────────────
CONN_STR = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
EVENTHUB_CONN_STR = os.environ["EVENTHUB_CONNECTION_STRING"]
EVENTHUB_NAME = os.environ.get("EVENTHUB_NAME_EVENTS", "user-events")
SPEED_FACTOR = float(os.environ.get("SPEED_FACTOR", "1000"))
# SPEED_FACTOR=1000 signifie qu'on rejoue 1000x plus vite que temps réel

# ── 1. Charger events depuis Blob Storage ────────────────────────────────
print("📥 Chargement events.csv depuis Blob Storage...")
blob_client = BlobServiceClient.from_connection_string(CONN_STR)
blob = blob_client.get_blob_client("datasets", "retail-rocket/events.csv")
data = blob.download_blob().readall()
df = pd.read_csv(io.BytesIO(data))
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"✅ {len(df):,} events chargés et triés par timestamp")

# ── 2. Connexion Event Hubs ───────────────────────────────────────────────
producer = EventHubProducerClient.from_connection_string(
    conn_str=EVENTHUB_CONN_STR,
    eventhub_name=EVENTHUB_NAME
)

# ── 3. Rejouer les events ────────────────────────────────────────────────
print(f"🚀 Envoi des events dans Event Hubs (vitesse x{SPEED_FACTOR})...")
prev_ts = None
batch_size = 0
total_sent = 0

with producer:
    event_data_batch = producer.create_batch()

    for _, row in df.iterrows():
        # Simuler le délai entre events
        if prev_ts is not None:
            delta = (row["timestamp"] - prev_ts) / (1000 * SPEED_FACTOR)
            if delta > 0:
                time.sleep(min(delta, 1.0))

        prev_ts = row["timestamp"]

        # Construire le message
        event = {
            "timestamp": int(row["timestamp"]),
            "visitorid": int(row["visitorid"]),
            "event": row["event"],
            "itemid": int(row["itemid"]),
        }

        try:
            event_data_batch.add(EventData(json.dumps(event)))
            batch_size += 1
        except ValueError:
            # Batch plein, on envoie et on recrée
            producer.send_batch(event_data_batch)
            total_sent += batch_size
            if total_sent % 10000 == 0:
                print(f"   📤 {total_sent:,} events envoyés...")
            event_data_batch = producer.create_batch()
            event_data_batch.add(EventData(json.dumps(event)))
            batch_size = 1

    # Envoyer le dernier batch
    if batch_size > 0:
        producer.send_batch(event_data_batch)
        total_sent += batch_size

print(f"✅ {total_sent:,} events envoyés dans Event Hubs")