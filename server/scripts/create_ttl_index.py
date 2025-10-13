#!/usr/bin/env python3
"""Create a TTL index on the conversations collection to auto-expire documents after 7 days.

Usage:
  source .venv/bin/activate
  python scripts/create_ttl_index.py

Requires MONGO_URI in env or .env.
"""
import os
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

env_file_path = os.getenv("ENV_FILE_PATH")
if env_file_path and os.path.exists(env_file_path):
  load_dotenv(env_file_path)
elif os.getenv("ENV", "").lower() != "production":
  try:
    load_dotenv()
  except Exception:
    pass

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
  raise SystemExit("MONGO_URI not set. Export MONGO_URI in your environment or provide an ENV_FILE_PATH for production.")

DB_NAME = os.getenv("MONGO_DB", "elearning-ai")
COLLECTION = os.getenv("CONVERSATIONS_COLLECTION", "conversations")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
coll = db[COLLECTION]

# TTL index on the 'timestamp' field; expireAfterSeconds = 7 days
expire_seconds = 7 * 24 * 60 * 60

print(f"Creating TTL index on {DB_NAME}.{COLLECTION} (expireAfterSeconds={expire_seconds})")
coll.create_index([("timestamp", ASCENDING)], expireAfterSeconds=expire_seconds)
print("Index created (or already exists).")
client.close()
