#!/usr/bin/env python3
"""Build course embeddings and write them to server/data/course_embeddings.json.

Run this manually from the repo root:
  cd server
  python scripts/build_course_embeddings.py

This script uses sentence-transformers to compute embeddings for each
course using the same build_course_text() logic as the server.
"""
import os
import json
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("Missing dependency: sentence-transformers. Install with: pip install sentence-transformers")
    raise


ROOT = os.path.dirname(os.path.dirname(__file__))
COURSES_PATH = os.path.join(ROOT, "courses.json")
OUT_DIR = os.path.join(ROOT, "data")
OUT_PATH = os.path.join(OUT_DIR, "course_embeddings.json")


def build_course_text(course: dict) -> str:
    title = course.get("title", "")
    desc = course.get("description", "")
    tags = ", ".join(course.get("tags", []))
    profession = ", ".join(course.get("profession", []))
    return f"Title: {title}, Description: {desc}, Tags: {tags}, Profession: {profession}"


def main():
    if not os.path.exists(COURSES_PATH):
        print("courses.json not found at", COURSES_PATH)
        return

    with open(COURSES_PATH, "r") as f:
        courses = json.load(f)

    texts = [build_course_text(c) for c in courses if c.get("id")]
    ids = [c.get("id") for c in courses if c.get("id")]

    print(f"Computing embeddings for {len(texts)} courses using all-MiniLM-L6-v2...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_docs = []
    for cid, course, emb in zip(ids, courses, embeddings):
        out_docs.append({
            "courseId": cid,
            "embedding": emb.tolist(),
            "title": course.get("title", ""),
            "tags": course.get("tags", []),
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "generated_at": datetime.utcnow().isoformat() + "Z",
        })

    with open(OUT_PATH, "w") as f:
        json.dump(out_docs, f, indent=2)

    print("Wrote embeddings to", OUT_PATH)


if __name__ == "__main__":
    main()
