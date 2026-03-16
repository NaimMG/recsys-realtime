#!/bin/bash
# Script to download datasets and upload to Azure Blob Storage

echo "📦 Downloading Retail Rocket dataset..."
kaggle datasets download -d retailrocket/ecommerce-dataset \
  -p ~/datasets/retail-rocket --unzip

echo "☁️ Uploading to Azure Blob Storage..."
python3 /home/azureuser/recsys-realtime/scripts/upload_to_blob.py

echo "✅ Done"