import os
from azure.storage.blob import BlobServiceClient

conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
client = BlobServiceClient.from_connection_string(conn_str)
container = client.get_container_client("datasets")

files = [
    "/home/azureuser/datasets/retail-rocket/events.csv",
    "/home/azureuser/datasets/retail-rocket/category_tree.csv",
]

for f in files:
    blob_name = "retail-rocket/" + os.path.basename(f)
    print(f"Uploading {blob_name}...")
    with open(f, "rb") as data:
        container.upload_blob(name=blob_name, data=data, overwrite=True)
    print(f"✅ {blob_name} uploadé")

print("✅ Tous les fichiers uploadés sur Azure Blob Storage")