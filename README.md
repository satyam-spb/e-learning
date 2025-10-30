# E-Learning AI Platform

This is a full-stack e-learning platform that uses AI models to provide personalized quiz experiences, course recommendations, and a RAG (Retrieval-Augmented Generation) powered chatbot. The system is built with FastAPI for the backend, MongoDB for the database, and hosted AI services for the core intelligence.

## 🚀 Key Features

- **Intelligent Quiz Generation:** Dynamically generates quizzes on demand based on a user-specified topic and difficulty.
- **Personalized Recommendations:** Recommends courses with a calculated "relevance" score based on the user's quiz performance, profession, and preferences.
- **RAG-Powered Chatbot:** A conversational AI assistant that uses a user's quiz history to provide context-aware, accurate answers.
- **Secure & Scalable:** Implemented with cybersecurity middleware to prevent access from malicious IP addresses.
- **Single-Service Architecture:** The entire backend is consolidated into a single, easy-to-manage FastAPI application.

---

## 🛠️ Technology Stack

- **Backend:** FastAPI
- **Database:** MongoDB
- **AI Models:** Groq (Llama 3.1) for LLM, Sentence-Transformers (MiniLM) for embeddings
- **Python Libraries:** `langchain`, `uvicorn`, `pymongo`, `requests`, `slowapi`
- **Frontend:** HTML, CSS, JavaScript (for testing purposes) || ReactJs for production

---

## 📁 Project Structure

The project follows a modular structure to keep the codebase clean and organized.

e-learning-platform/
├── .env <-- Your API keys and secrets go here
├── .gitignore <-- A good practice to keep your repo clean
├── README.md <-- The documentation for the project
├── client-test/ <-- Your frontend test files
│ └── index.html
└── server/ <-- Your backend code lives here
├── app.py <-- The main backend application

# E-Learning AI Platform

A compact, deployable e-learning platform using FastAPI (backend), MongoDB (persistence), sentence-transformers for embeddings, and a hosted LLM for conversational features.

This README contains copy/paste-ready commands for macOS and Windows to get the project running, precompute embeddings, seed the database, and run quick smoke tests.

---

## What's included / recent changes

- Precomputed-embeddings workflow: `server/scripts/build_course_embeddings.py` writes `server/data/course_embeddings.json`.
- Backend prefers local JSON embeddings (EMB_MAP). Heavy compute at startup is opt-in via `EMBEDDINGS_SOURCE=compute`.
- Added `/status` endpoint for health and counts.
- CORS is configurable via `ALLOWED_ORIGINS` env var (comma-separated).
- `client/vite.config.ts` base set to `/` for Vercel compatibility.
- `server/requirements-pinned.txt` added for deterministic installs.

---

## Prerequisites

- Python 3.11+ (3.13 was used while developing here)
- Git
- MongoDB (local or remote)

For the local embedding build you'll need the Python packages in `server/requirements-pinned.txt`. The MiniLM model is lightweight; a standard dev laptop is sufficient.

---

## macOS setup (quick)

1. Activate the included server venv (recommended):

```bash
# from repo root
source server/venv/bin/activate
```

Or create a fresh venv:

```bash
python3 -m venv server/venv
source server/venv/bin/activate
```

2. Install pinned dependencies:

```bash
pip install -r server/requirements-pinned.txt
```

---

## Windows setup (PowerShell)

1. Activate included server venv (if present):

```powershell
# from repo root
.\server\venv\Scripts\Activate.ps1
```

Or create a fresh venv:

```powershell
python -m venv server\venv
.\server\venv\Scripts\Activate.ps1
```

2. Install pinned dependencies:

```powershell
pip install -r server\requirements-pinned.txt
```

---

## Environment variables

In production, set secrets and configuration as real environment variables (for example in your host or CI settings). Do NOT rely on a `.env` file in production.

For local development you can create a `.env` file in the repo root. The app will automatically load it unless `ENV=production` is set.

Example `.env` (development only — do not commit):

```env
GROQ_API_KEY="your_groq_api_key_here"
ABUSEIPDB_API_KEY="your_abuseipdb_api_key_here"
MONGO_URI="mongodb://localhost:27017/"
ALLOWED_ORIGINS="*"
EMBEDDINGS_SOURCE="local_json"
```

- `ALLOWED_ORIGINS` is comma-separated (e.g. `https://your-frontend.vercel.app,https://stats.uptimerobot.com,*`).
- `EMBEDDINGS_SOURCE` defaults to `local_json`. Use `compute` only when you want server-side embedding computation.

To run with production-style environment variables locally, prefix your run command with `ENV=production` so the app won't auto-load `.env`.

Example (macOS / Linux):

```bash
ENV=production uvicorn app:app --host 0.0.0.0 --port 8000
```

### Frontend (Vite) environment variables

The frontend reads runtime config from Vite env variables prefixed with `VITE_`. In production, set these in your host (Vercel, Netlify, etc.) rather than relying on a `.env` file.

Example variables to set in production:

```env
VITE_API_BASE=https://your-backend.example.com
VITE_FIREBASE_API_KEY=...
VITE_FIREBASE_AUTH_DOMAIN=...
VITE_FIREBASE_PROJECT_ID=...
```

Local development may use `client/.env` (this repo includes `client/.env.example` to copy from).

### Optional production env-file (ENV_FILE_PATH)

If your hosting environment provides a file with environment variables, you can point the server to it using `ENV_FILE_PATH` (absolute path). Behavior:

- If `ENV_FILE_PATH` is set and the file exists, the server will load variables from that file (useful for container-based deployments that drop a file).
- Otherwise, when `ENV` is not set to `production`, the server will attempt to load a local `.env` for developer convenience.
- If `ENV=production` and `ENV_FILE_PATH` is not provided (or file missing), the server will not read any `.env` file and will rely on the environment provided by the host.

---

## Build precomputed embeddings (one-off, recommended)

Precomputing embeddings avoids heavy startup work and is recommended for free-tier deployments.

macOS / Linux:

```bash
source server/venv/bin/activate
cd server
python scripts/build_course_embeddings.py
```

Windows (PowerShell):

```powershell
.\server\venv\Scripts\Activate.ps1
cd server
python scripts\build_course_embeddings.py
```

After running you'll get `server/data/course_embeddings.json`. The server prefers this file automatically when `EMBEDDINGS_SOURCE=local_json`.

---

## Seed quizzes (manual)

Load quiz data into MongoDB (idempotent):

```bash
cd server
python load_quizzes.py
```

---

## Run the server (development)

With the server venv active and from the `server/` folder:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The server will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## Quick smoke tests

- GET `/status` — returns counts and `embeddings_source`.
- POST `/recommend` — JSON: `{ "user_id": "id", "profession": "Developer", "preferences": ["python","backend"] }`.

Example (macOS):

```bash
curl -sS http://127.0.0.1:8000/status | python3 -m json.tool
curl -sS -X POST http://127.0.0.1:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"test-user","profession":"Developer","preferences":["python","backend"]}' | python3 -m json.tool
```

---

## Deployment notes

- Keep `EMBEDDINGS_SOURCE=local_json` for quick cold starts in Render or similar.
- If the server should compute embeddings at startup, set `EMBEDDINGS_SOURCE=compute` (requires sufficient resources).
- Set `ALLOWED_ORIGINS` to your frontend domain(s) and any monitoring origins (e.g., UptimeRobot) for production.
- `client/vite.config.ts` base is set to `/` for Vercel compatibility.

---

If you'd like I can commit this README and the `client/vite.config.ts` update. Also tell me if you want `server/data/course_embeddings.json` committed to the repository (I left the generated file uncommitted by default).
