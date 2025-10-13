import os
import json
from typing import List, Dict, Optional, Callable, Any
try:
    from dotenv import load_dotenv
except Exception:
    # If python-dotenv is not installed, provide a no-op loader so the app can still run.
    def load_dotenv(path: str = None):
        # no-op fallback when python-dotenv isn't available
        return False
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime
from pymongo import MongoClient, ASCENDING
import numpy as np
# from sentence_transformers import SentenceTransformer
# sentence_transformers is imported lazily inside lazy_load_embedding_model()
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
from slowapi.util import get_remote_address
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment loading strategy:
# - If ENV_FILE_PATH is set and the file exists, load it (useful for hosts that drop a file into the container).
# - Else if ENV != 'production', load local .env for developer convenience.
# - Otherwise (production, and no ENV_FILE_PATH), do not load .env and rely on true environment variables.
env_file_path = os.getenv("ENV_FILE_PATH")
if env_file_path:
    if os.path.exists(env_file_path):
        try:
            load_dotenv(env_file_path)
            logger.info(f"Loaded environment from ENV_FILE_PATH={env_file_path}")
        except Exception:
            logger.warning(f"Failed to load env file at ENV_FILE_PATH={env_file_path}; proceeding with environment variables.")
    else:
        logger.warning(f"ENV_FILE_PATH is set to '{env_file_path}' but file does not exist. Not loading it.")
else:
    if os.getenv("ENV", "").lower() != "production":
        try:
            load_dotenv()
            logger.info("Loaded local .env for development (ENV != production).")
        except Exception:
            logger.debug("Could not load local .env — proceeding with existing environment variables.")

# --- API Key Check and Database Setup ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment variables. Set GROQ_API_KEY to enable chat/LLM features.")

ABUSEIPDB_API_KEY = os.getenv("ABUSEIPDB_API_KEY")
if not ABUSEIPDB_API_KEY:
    # AbuseIPDB is optional. If not provided, IP threat checks will be skipped.
    logger.warning("ABUSEIPDB_API_KEY not set; skipping AbuseIPDB checks (recommended for production).")
    ABUSEIPDB_API_KEY = None

MONGO_URI = os.getenv("MONGO_URI") or "mongodb://localhost:27017/"
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["elearning-ai"]
quiz_collection = db["quizzes"]
course_embeddings_collection = db["courseEmbeddings"]
user_results_collection = db["user_quiz_results"]
conversations_collection = db["conversations"]

# Add connection check logic
try:
    mongo_client.admin.command('ping')
    logger.info("Successfully connected to MongoDB.")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")

# Embeddings source control: prefer DB by default for deployed environments
# EMBEDDINGS_SOURCE can be: local_json | compute | db
EMBEDDINGS_SOURCE = os.getenv("EMBEDDINGS_SOURCE", "db")
embedding_model = None

def lazy_load_embedding_model():
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

# Setup Groq client if LLM is enabled and API key exists
llm = None
try:
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)
    logger.info("Initialized ChatGroq LLM client.")
except Exception as e:
    # Fail fast in this config: chat must be available
    raise RuntimeError(f"Failed to initialize ChatGroq LLM client: {e}")

# --- FastAPI App ---
app = FastAPI()

# Developer-enforced system instruction: assistant must only answer tech/CS questions
SYSTEM_INSTRUCTION = (
    "You are an AI assistant whose scope is technology and computer science only. "
    "Always follow this rule. If the user asks about topics outside technology or computer science, "
    "politely decline and say the question is out of scope. Provide a brief suggestion for where the user "
    "might find help (for example, suggest a general knowledge source) but do NOT answer non-technical questions."
)


@app.on_event("startup")
async def ensure_ttl_index():
    """Ensure a TTL index exists on conversations.timestamp (7 days). This is idempotent."""
    try:
        expire_seconds = 7 * 24 * 60 * 60
        conversations_collection.create_index([("timestamp", ASCENDING)], expireAfterSeconds=expire_seconds)
        logger.info(f"Ensured TTL index on conversations.timestamp (expireAfterSeconds={expire_seconds})")
    except Exception as e:
        logger.warning(f"Could not create TTL index on conversations collection: {e}")

# Add the cybersecurity middleware
ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"
@app.middleware("http")
async def check_threat_ip_middleware(request: Request, call_next: Callable):
    client_ip = get_remote_address(request)
    
    if client_ip in ["127.0.0.1", "::1"]:
        return await call_next(request)

    # If AbuseIPDB is not configured, skip this check
    if not ABUSEIPDB_API_KEY:
        return await call_next(request)

    try:
        headers = {"Key": ABUSEIPDB_API_KEY, "Accept": "application/json"}
        params = {"ipAddress": client_ip, "maxAgeInDays": 90}
        response = requests.get(ABUSEIPDB_URL, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        result = response.json()
        data = result.get("data", {})
        score = data.get("abuseConfidenceScore", 0)

        if score > 50:
            raise HTTPException(status_code=403, detail="Request blocked: Malicious IP detected")
        
        return await call_next(request)
    except requests.RequestException as e:
        logger.warning(f"Failed to check IP with AbuseIPDB: {e}")
        return await call_next(request)

# CORS configuration (no hard-coded URLs in code)
# Behavior:
#  - If ALLOWED_ORIGINS is set (comma-separated), it will be used verbatim.
#  - Otherwise, the app will look for FRONTEND_URL and UPTIME_ROBOT_URL environment variables and use them.
#  - If none of these are provided, the server will log a warning and fall back to allowing all origins ("*") to avoid accidental blocking in development.
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
front_end_url = os.getenv("FRONTEND_URL")
uptime_robot_url = os.getenv("UPTIME_ROBOT_URL")

if allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    allowed_origins = []
    if front_end_url:
        allowed_origins.append(front_end_url.rstrip("/"))
    if uptime_robot_url:
        allowed_origins.append(uptime_robot_url.rstrip("/"))

if not allowed_origins:
    logger.warning("No CORS origins configured (ALLOWED_ORIGINS, FRONTEND_URL, UPTIME_ROBOT_URL are all empty). Falling back to allow_origins=['*']. Consider setting FRONTEND_URL in your environment or .env.")
    allowed_origins = ["*"]

logger.info(f"Configured CORS allow_origins={allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class Quiz(BaseModel):
    score: float
    course: str

class RecommendationRequest(BaseModel):
    user_id: str
    profession: str
    preferences: List[str]

class ChatInput(BaseModel):
    message: str
    conversation_id: str
    user_id: Optional[str] = ""

class UserQuizResult(BaseModel):
    user_id: str
    topic: str
    score: float
    profession: Optional[str] = None

class QuizQuestion(BaseModel):
    id: str
    question: str
    options: List[str]
    correctAnswer: int
    weight: int
    topic: Optional[str]

# --- Helper Functions ---
def build_course_text(course: dict) -> str:
    title = course.get("title", "")
    desc = course.get("description", "")
    tags = ", ".join(course.get("tags", []))
    profession = ", ".join(course.get("profession", []))
    return f"Title: {title}, Description: {desc}, Tags: {tags}, Profession: {profession}"

def cache_course_embeddings(courses: List[dict]):
    """Helper for caching embeddings (synchronous)."""
    logger.info(f"Caching embeddings for {len(courses)} courses.")
    model = lazy_load_embedding_model()
    for course in courses:
        course_text = build_course_text(course)
        try:
            embedding = model.encode([course_text], show_progress_bar=False)[0]
        except Exception as e:
            logger.error(f"Failed to encode course '{course.get('title', '')}': {e}")
            continue
        course_id = course.get("id")
        if not course_id:
            logger.warning(f"Skipping course with missing ID: {course.get('title', 'Unknown')}")
            continue
        course_embeddings_collection.update_one(
            {"courseId": course_id},
            {"$set": {"embedding": embedding.tolist(), "title": course.get("title", ""), "tags": course.get("tags", [])}},
            upsert=True
        )
    logger.info("Course embeddings cached successfully.")

def ensure_course_embeddings_cached(courses: List[dict]):
    """Check cache status and generate embeddings if needed (synchronous)."""
    if not courses:
        logger.warning("[startup] Static course list is empty. Skipping embedding generation.")
        return
    
    try:
        existing_count = course_embeddings_collection.count_documents({})
    except Exception as exc:
        logger.error(f"[startup] Warning: could not count courseEmbeddings: {exc}")
        existing_count = 0

    if existing_count >= len(courses):
        logger.info(f"[startup] Skipping embedding generation – {existing_count} embeddings already cached.")
        return

    # Only compute embeddings at startup if explicitly allowed
    if EMBEDDINGS_SOURCE == "compute":
        logger.info("[startup] Computing course embeddings (EMBEDDINGS_SOURCE=compute)...")
        cache_course_embeddings(courses)
        logger.info("[startup] Course embeddings cached successfully.")
    else:
        logger.info("[startup] EMBEDDINGS_SOURCE != 'compute'. Skipping compute. Expecting local JSON or DB entries.")

try:
    # Load courses.json from the server folder (path relative to this file)
    courses_path = os.path.join(os.path.dirname(__file__), "courses.json")
    with open(courses_path, "r") as f:
        COURSES = json.load(f)
except Exception as e:
    # Log a friendly warning; downstream code expects COURSES to be a list.
    logger.warning(f"[startup] Warning: could not load courses.json: {e}")
    COURSES = []

# --- Optionally load precomputed embeddings from server/data/course_embeddings.json when requested
EMB_JSON_PATH = os.path.join(os.path.dirname(__file__), "data", "course_embeddings.json")
EMB_MAP: Dict[str, Any] = {}
if EMBEDDINGS_SOURCE == "local_json":
    if os.path.exists(EMB_JSON_PATH):
        try:
            with open(EMB_JSON_PATH, "r") as f:
                emb_docs = json.load(f)
            for doc in emb_docs:
                cid = doc.get("courseId")
                if cid and "embedding" in doc:
                    EMB_MAP[cid] = np.array(doc["embedding"])
            logger.info(f"Loaded {len(EMB_MAP)} embeddings from {EMB_JSON_PATH}")
        except Exception as e:
            logger.error(f"Failed to load embeddings from JSON: {e}")
    else:
        logger.info("No local embeddings JSON found; EMB_MAP is empty.")
else:
    logger.info(f"EMBEDDINGS_SOURCE={EMBEDDINGS_SOURCE}; preferring DB embeddings if available.")


def infer_course_difficulty(course: dict) -> str:
    """Simple heuristic to infer course difficulty from title or tags.
    Returns one of: 'easy', 'medium', 'hard'.
    """
    title = course.get("title", "").lower()
    tags = [t.lower() for t in course.get("tags", [])]
    easy_keywords = ["intro", "beginner", "basics", "fundamentals", "getting started"]
    hard_keywords = ["advanced", "expert", "deep", "masterclass", "in-depth"]

    # If explicit difficulty tag exists, respect it
    diff_tag = course.get("difficulty")
    if diff_tag:
        dt = str(diff_tag).lower()
        if dt in ("easy", "medium", "hard"):
            return dt

    # Check title and tags heuristics
    if any(k in title for k in hard_keywords) or any(k in " ".join(tags) for k in hard_keywords):
        return "hard"
    if any(k in title for k in easy_keywords) or any(k in " ".join(tags) for k in easy_keywords):
        return "easy"
    return "medium"


# Conversation helpers
def store_conversation_message(conversation_id: str, user_id: Optional[str], role: str, content: str):
    doc = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow()
    }
    try:
        conversations_collection.insert_one(doc)
    except Exception as e:
        logger.error(f"Failed to store conversation message: {e}")


def get_conversation_messages(conversation_id: str, limit: int = 100) -> List[dict]:
    try:
        msgs = list(conversations_collection.find({"conversation_id": conversation_id}).sort("timestamp", 1).limit(limit))
        return msgs
    except Exception as e:
        logger.error(f"Failed to fetch conversation messages: {e}")
        return []


async def summarize_and_compact_conversation(conversation_id: str, keep_last: int = 20) -> Optional[str]:
    """Summarize the older part of the conversation into a short summary, insert it as a system message,
    and remove the older messages except the last `keep_last` messages. Returns the summary string or None."""
    try:
        all_msgs = list(conversations_collection.find({"conversation_id": conversation_id}).sort("timestamp", 1))
        if len(all_msgs) <= keep_last + 5:
            return None

        # Messages to summarize: everything except the last `keep_last`
        to_summarize = all_msgs[:-keep_last]
        formatted = []
        for m in to_summarize:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            formatted.append(f"{role.upper()}: {content}")
        messages_text = "\n".join(formatted)

        # If LLM features are not available, skip summarization.
        if not (llm is not None):
            logger.info("LLM not available; skipping conversation summarization.")
            return None

        summarize_prompt = PromptTemplate(
            input_variables=["messages"],
            template=(
                f"{SYSTEM_INSTRUCTION}\n\n"
                "Summarize the following conversation between the user and assistant into a short 2-3 sentence summary. "
                "Preserve important facts, user goals, and any decisions. Keep it concise.\n\nConversation:\n{messages}\n\nSummary:"
            )
        )
        chain = summarize_prompt | llm | StrOutputParser()
        summary = await chain.ainvoke({"messages": messages_text})
        summary_text = str(summary).strip()

        # Insert the summary as a system message
        store_conversation_message(conversation_id, None, "system", f"Conversation summary: {summary_text}")

        # Remove the older messages that were summarized to avoid growing DB too large
        # Keep only the last `keep_last` messages plus the newly inserted summary
        try:
            # find timestamps to keep
            last_msgs = all_msgs[-keep_last:]
            last_timestamps = [m["timestamp"] for m in last_msgs]
            oldest_to_keep = min(last_timestamps) if last_timestamps else datetime.utcnow()
            conversations_collection.delete_many({"conversation_id": conversation_id, "timestamp": {"$lt": oldest_to_keep}})
        except Exception as e:
            logger.warning(f"Failed to prune old messages after summarization: {e}")

        return summary_text
    except Exception as e:
        logger.error(f"Error during conversation summarization: {e}")
        return None

@app.get("/quiz", response_model=List[QuizQuestion])
async def get_quiz(
    difficulty: str = Query(...),
    topic: Optional[str] = Query(None)
):
    difficulty_mapping = {"easy": 1, "medium": 2, "hard": 3}
    if difficulty.lower() not in difficulty_mapping:
        raise HTTPException(status_code=400, detail="Invalid difficulty.")
        # If LLM features or prompt tooling are not available, skip summarization.
        if not globals().get('LLM_ENABLED', False) or not globals().get('LANGCHAIN_CORE_OK', False):
            logger.info("LLM or prompt tooling not available; skipping conversation summarization.")
            return None

    try:
        questions = list(quiz_collection.aggregate([
            {"$match": match_query},
            {"$sample": {"size": 10}}
        ]))
        if not questions:
            raise HTTPException(status_code=404, detail=f"No questions found for difficulty '{difficulty}' and topic '{topic or 'any'}'.")
        return questions
    except Exception as e:
        logger.error(f"Error fetching quiz questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching quiz: {str(e)}")

@app.get("/quiz/topics")
async def get_quiz_topics():
    try:
        topics = quiz_collection.distinct("topic")
        return {"topics": topics}
    except Exception as e:
        logger.error(f"Error fetching quiz topics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching topics: {str(e)}")


@app.get("/courses")
async def get_courses():
    """Return the list of courses loaded from courses.json.
    This endpoint is used by the frontend to populate profession and topic selects.
    """
    try:
        return COURSES
    except Exception as e:
        logger.error(f"Error returning courses: {e}")
        raise HTTPException(status_code=500, detail="Failed to load courses")

@app.post("/quiz/results")
async def save_quiz_results(result: UserQuizResult):
    try:
        # Store user's quiz result along with optional profession
        user_results_collection.update_one(
            {"user_id": result.user_id},
            {"$set": result.dict()},
            upsert=True
        )
        logger.info(f"Saved quiz results for user {result.user_id} on topic {result.topic}.")
        return {"message": "Quiz results saved successfully."}
    except Exception as e:
        logger.error(f"Error saving quiz results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving quiz results: {str(e)}")

@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    try:
        logger.info(f"Processing recommendation request for user_id: {request.user_id}, profession: {request.profession}, preferences: {request.preferences}")

        # Fetch user quiz results
        user_quiz = None
        user_quiz_result = user_results_collection.find_one({"user_id": request.user_id})
        if user_quiz_result:
            user_quiz = Quiz(score=user_quiz_result["score"], course=user_quiz_result["topic"])
            logger.info(f"Found quiz result: topic={user_quiz.course}, score={user_quiz.score}")

        user_prefs = set(p.strip().lower() for p in request.preferences if p.strip())  # Remove empty preferences
        logger.info(f"Normalized user preferences: {user_prefs}")

        # --- PRIMARY FILTER: Filter courses by quiz topic (if available) ---
        filtered_courses = COURSES
        if user_quiz and user_quiz.course:
            topic_filter = user_quiz.course.lower().strip()
            logger.info(f"Applying topic filter: {topic_filter}")
            filtered_courses = [
                c for c in COURSES
                if topic_filter in c.get("title", "").lower() or 
                   topic_filter in c.get("description", "").lower() or
                   topic_filter in [t.lower() for t in c.get("tags", [])]
            ]
            logger.info(f"After topic filter, found {len(filtered_courses)} courses.")

        # --- SECONDARY FILTER: Apply user preferences and profession ---
        if user_prefs or request.profession:
            logger.info("Applying preferences and profession filter")
            filtered_courses = [
                c for c in filtered_courses 
                if (
                    (not user_prefs or any(p in [t.lower() for t in c.get("tags", [])] for p in user_prefs)) and
                    (not request.profession or request.profession.lower() in [p.lower() for p in c.get("profession", [])])
                )
            ]
            logger.info(f"After preferences/profession filter, found {len(filtered_courses)} courses.")

        # JavaScript exclusion for Java quiz
        if user_quiz and user_quiz.course.lower() == "java":
            filtered_courses = [c for c in filtered_courses if "javascript" not in [t.lower() for t in c.get("tags", [])]]
            logger.info(f"Applied JavaScript exclusion, found {len(filtered_courses)} courses.")

        # Fallback to broader filtering if too few courses remain
        if len(filtered_courses) < 5:  # Arbitrary threshold to ensure enough courses
            logger.warning("Too few courses after filtering. Applying broader filter.")
            filtered_courses = [
                c for c in COURSES
                if (
                    (not user_prefs or any(p in [t.lower() for t in c.get("tags", [])] for p in user_prefs)) or
                    (not request.profession or request.profession.lower() in [p.lower() for p in c.get("profession", [])])
                )
            ]
            logger.info(f"After broader filter, found {len(filtered_courses)} courses.")

        if not filtered_courses:
            logger.error("No courses available after all filters.")
            raise HTTPException(status_code=200, detail="No courses available. Please try broadening your preferences or check course data.")

        # --- Generate user embedding ---
        quiz_part = f"Quiz: {user_quiz.course} with score {user_quiz.score}%" if user_quiz and user_quiz.course else "No quiz data"
        pref_part = f"Preferences: {', '.join(user_prefs)}" if user_prefs else "No preferences"
        prof_part = f"Profession: {request.profession}" if request.profession else "No profession"
        user_text = f"{prof_part}. {pref_part}. {quiz_part}."
        logger.info(f"User text for embedding: {user_text}")
        # Create user embedding using lazy model loading if necessary
        if EMBEDDINGS_SOURCE == "local_json":
            # still need an embedding model for user text; lazy-load it
            model = lazy_load_embedding_model()
            user_embedding = model.encode([user_text], normalize_embeddings=True)[0]
        else:
            model = lazy_load_embedding_model()
            user_embedding = model.encode([user_text], normalize_embeddings=True)[0]

        # --- Retrieve course embeddings ---
        # Prepare embedding map: prefer local JSON map (EMB_MAP) else fall back to DB
        emb_map = {}
        if EMB_MAP:
            emb_map = EMB_MAP
            logger.info(f"Using {len(emb_map)} embeddings from local JSON map.")
        else:
            course_ids = [c["id"] for c in filtered_courses if c.get("id")]
            course_embeddings_docs = list(course_embeddings_collection.find({"courseId": {"$in": course_ids}}))
            logger.info(f"Retrieved {len(course_embeddings_docs)} course embeddings from DB.")
            emb_map = {doc["courseId"]: np.array(doc["embedding"]) for doc in course_embeddings_docs if "embedding" in doc}

        aligned_filtered_courses = [c for c in filtered_courses if c.get("id") in emb_map]
        logger.info(f"Aligned courses with embeddings: {len(aligned_filtered_courses)}")

        if not aligned_filtered_courses:
            logger.error("No courses with valid embeddings found.")
            raise HTTPException(status_code=200, detail="No courses with valid embeddings found. Try broadening your preferences or ensure embeddings are cached.")

        embeddings_list = [emb_map[c["id"]] for c in aligned_filtered_courses]

        # --- Compute similarities ---
        try:
            user_embedding_2d = user_embedding.reshape(1, -1)
            embeddings_array = np.array(embeddings_list)
            if embeddings_array.ndim == 1 and embeddings_array.size > 0:
                embeddings_array = embeddings_array.reshape(1, -1)
            similarities = cosine_similarity(user_embedding_2d, embeddings_array)[0]
            logger.info(f"Computed similarities for {len(similarities)} courses: {similarities}")
        except ValueError as ve:
            logger.error(f"Embedding processing error: {str(ve)}")
            raise HTTPException(status_code=500, detail="Recommendation processing error: Corrupted embedding data in DB. Please clear the 'courseEmbeddings' collection and restart the server.")

        # --- Apply relevance calculation based on quiz score and difficulty ---
        user_proficiency = "none"
        if user_quiz and user_quiz.score is not None:
            user_proficiency = "high" if user_quiz.score >= 80 else "medium" if user_quiz.score >= 50 else "low"
            logger.info(f"User proficiency: {user_proficiency}")

        relevance_scores = []
        for i, course in enumerate(aligned_filtered_courses):
            semantic_score = similarities[i]
            course_difficulty = infer_course_difficulty(course)
            difficulty_multiplier = 1.0
            if user_proficiency == "high" and course_difficulty in ["easy", "medium"]:
                difficulty_multiplier = 1.1
            elif user_proficiency == "medium" and course_difficulty == "medium":
                difficulty_multiplier = 1.2
            elif user_proficiency == "low" and course_difficulty == "easy":
                difficulty_multiplier = 1.3
            elif user_proficiency == "high" and course_difficulty == "hard":
                difficulty_multiplier = 0.8
            elif user_proficiency == "low" and course_difficulty == "hard":
                difficulty_multiplier = 0.5
            # Add small randomization to break ties in similarity scores
            random_noise = np.random.uniform(-0.01, 0.01)
            final_relevance = min(1.0, (semantic_score * difficulty_multiplier + random_noise)) * 100
            relevance_scores.append(final_relevance)

        # Sort by relevance and select top 6
        sorted_indices = np.argsort(relevance_scores)[::-1]
        top_courses_with_relevance = []
        
        for i in sorted_indices[:8]:
            course = aligned_filtered_courses[i]
            relevance = int(relevance_scores[i])
            course_with_relevance = course.copy()
            course_with_relevance["relevance"] = relevance
            top_courses_with_relevance.append(course_with_relevance)

        logger.info(f"Returning {len(top_courses_with_relevance)} recommendations: {[c['title'] for c in top_courses_with_relevance]}")
        return {"recommendations": top_courses_with_relevance}

    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

@app.get("/status")
async def status():
    """Return a simple status with counts and embeddings source."""
    def safe_count(collection):
        try:
            return collection.count_documents({})
        except Exception:
            return None

    quizzes_count = safe_count(quiz_collection)
    results_count = safe_count(user_results_collection)

    # If EMB_MAP is populated (local JSON), use its length for embeddings_count
    embeddings_count = len(EMB_MAP) if EMB_MAP else safe_count(course_embeddings_collection)

    return {
        "status": "ok",
        "quizzes_count": quizzes_count,
        "embeddings_count": embeddings_count,
        "user_quiz_results_count": results_count,
        "embeddings_source": EMBEDDINGS_SOURCE,
        "llm_enabled": bool('llm' in globals() and llm is not None),
    }


@app.post("/chat/")
async def chat(input: ChatInput):
    try:
        # Ensure LLM is available for chat
        if llm is None:
            logger.error("Chat requested but LLM client is not initialized.")
            raise HTTPException(status_code=503, detail="Chat/LLM functionality is not available — server misconfiguration.")

        # Persist the incoming user message
        store_conversation_message(input.conversation_id, input.user_id, "user", input.message)

        # Optionally fetch user's saved quiz results for context and include as system message on first interaction
        user_quiz_results = None
        if input.user_id:
            user_quiz_results = user_results_collection.find_one({"user_id": input.user_id})
            logger.info(f"User quiz results: {user_quiz_results}")
            # If this is the first message (no prior messages), insert system note about quiz/profession
            existing = conversations_collection.count_documents({"conversation_id": input.conversation_id})
            if existing <= 1 and user_quiz_results:
                prof = user_quiz_results.get("profession") or "Unknown"
                score = user_quiz_results.get("score")
                topic = user_quiz_results.get("topic")
                system_text = f"User quiz result: topic={topic}, score={score}%. Profession: {prof}."
                store_conversation_message(input.conversation_id, input.user_id, "system", system_text)

        # Fetch conversation history
        history = get_conversation_messages(input.conversation_id, limit=500)

        # Summarize older history if it's too long
        if len(history) > 60:
            summary = await summarize_and_compact_conversation(input.conversation_id, keep_last=25)
            if summary:
                logger.info(f"Summarized conversation: {summary}")
            # Reload history after summarization
            history = get_conversation_messages(input.conversation_id, limit=500)

        # Build context from last messages (skip system summaries if present?)
        context_parts = []
        # include up to last 25 messages
        recent = history[-25:]
        for m in recent:
            role = m.get("role")
            content = m.get("content")
            if role and content:
                context_parts.append(f"{role}: {content}")
        context = "\n".join(context_parts)

        # Also include some relevant quiz questions for topical context (best-effort)
        query_terms = input.message.lower().split()
        topic_terms = [term for term in query_terms if term in ["java", "object", "class", "programming"]]
        topic_regex = "|".join([f".*{term}.*" for term in topic_terms]) if topic_terms else ""
        if topic_regex:
            retrieved_questions = list(quiz_collection.find({
                "$or": [
                    {"question": {"$regex": input.message, "$options": "i"}},
                    {"topic": {"$regex": topic_regex, "$options": "i"}}
                ]
            }).limit(5))
        else:
            retrieved_questions = list(quiz_collection.find({"question": {"$regex": input.message, "$options": "i"}}).limit(5))

        if retrieved_questions:
            context += "\nRelevant quiz questions for context:\n"
            for q in retrieved_questions:
                correct_option_index = q.get('correctAnswer')
                correct_answer = q['options'][correct_option_index] if 'options' in q and correct_option_index is not None else "N/A"
                context += f"- Question: {q.get('question', 'N/A')}\n  Correct Answer: {correct_answer}\n"

        # Build the prompt using conversation context
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                f"{SYSTEM_INSTRUCTION}\n\n"
                "You are a helpful AI assistant. Use the conversation context and relevant quiz info to answer the user's question concisely. "
                "Context: {context}\nUser's Question: {question}\nAnswer:"
            )
        )

        chain = prompt_template | llm | StrOutputParser()
        response = await chain.ainvoke({"context": context, "question": input.message})
        logger.info(f"Chat response generated: {str(response)[:100]}...")

        # Persist assistant response
        store_conversation_message(input.conversation_id, input.user_id, "assistant", str(response))

        return {"response": response, "conversation_id": input.conversation_id}

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat request: {str(e)}")


@app.get("/chat/history")
async def chat_history(conversation_id: str = Query(...)):
    try:
        msgs = get_conversation_messages(conversation_id, limit=1000)
        # Convert ObjectId and datetime to serializable forms
        out = []
        for m in msgs:
            ts = m.get("timestamp")
            ts_iso = ts.isoformat() if ts is not None else None
            out.append({
                "role": m.get("role"),
                "content": m.get("content"),
                "timestamp": ts_iso,
            })
        return {"messages": out}
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get chat history")


@app.delete("/chat/{conversation_id}")
async def delete_chat(conversation_id: str):
    try:
        result = conversations_collection.delete_many({"conversation_id": conversation_id})
        logger.info(f"Deleted {result.deleted_count} messages for conversation {conversation_id}")
        return {"deleted_count": int(result.deleted_count)}
    except Exception as e:
        logger.error(f"Failed to delete chat {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete chat history")


@app.post("/chat/cleanup")
async def chat_cleanup(payload: dict):
    conversation_id = payload.get("conversation_id")
    if not conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id is required")
    try:
        result = conversations_collection.delete_many({"conversation_id": conversation_id})
        logger.info(f"Cleanup deleted {result.deleted_count} messages for conversation {conversation_id}")
        return {"deleted_count": int(result.deleted_count)}
    except Exception as e:
        logger.error(f"Failed to cleanup chat {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup chat history")

# Serve static files from client-test folder
app.mount("/static", StaticFiles(directory="../client-test"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("../client-test/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
