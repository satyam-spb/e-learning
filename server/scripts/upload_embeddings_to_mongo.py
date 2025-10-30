#!/usr/bin/env python3
"""Upload server/data/course_embeddings.json into the MongoDB courseEmbeddings collection.

Usage:
  source .venv/bin/activate
  python scripts/upload_embeddings_to_mongo.py

Requires MONGO_URI in environment or in .env at repo root.
"""
import os
import json
from pymongo import MongoClient
from dotenv import load_dotenv

 # Environment loading strategy: respect ENV_FILE_PATH if provided, else load .env only when ENV != 'production'.
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
    raise SystemExit("MONGO_URI not set in environment. Export MONGO_URI or provide an ENV_FILE_PATH pointing to a file with it for production.")

DB_NAME = os.getenv("MONGO_DB", "elearning-ai")
COLLECTION = os.getenv("EMB_COLLECTION", "courseEmbeddings")

ROOT = os.path.dirname(os.path.dirname(__file__))
EMB_PATH = os.path.join(ROOT, "data", "course_embeddings.json")

if not os.path.exists(EMB_PATH):
    raise SystemExit(f"Embeddings file not found at {EMB_PATH}. Run scripts/build_course_embeddings.py first.")

with open(EMB_PATH, "r") as f:
    docs = json.load(f)

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
coll = db[COLLECTION]

count = 0
for doc in docs:
    cid = doc.get("courseId")
    emb = doc.get("embedding")
    if not cid or not emb:
        continue
    # Upsert the embedding document
    coll.update_one({"courseId": cid}, {"$set": {"courseId": cid, "embedding": emb, "title": doc.get("title",""), "tags": doc.get("tags",[]), "model": doc.get("model","") }}, upsert=True)
    count += 1

print(f"Upserted {count} embeddings into {DB_NAME}.{COLLECTION}")
client.close()
