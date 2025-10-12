import os
import json
from typing import List, Dict, Optional, Callable, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pymongo import MongoClient
import numpy as np
from sentence_transformers import SentenceTransformer
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

load_dotenv()

# --- API Key Check and Database Setup ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file.")

ABUSEIPDB_API_KEY = os.getenv("ABUSEIPDB_API_KEY")
if not ABUSEIPDB_API_KEY:
    raise ValueError("ABUSEIPDB_API_KEY is not set in the .env file.")

MONGO_URI = os.getenv("MONGO_URI") or "mongodb://localhost:27017/"
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["elearning-ai"]
quiz_collection = db["quizzes"]
course_embeddings_collection = db["courseEmbeddings"]
user_results_collection = db["user_quiz_results"]

# Add connection check logic
try:
    mongo_client.admin.command('ping')
    logger.info("Successfully connected to MongoDB.")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")

# Load embedding model once for RAG and recommendations
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Setup Groq client
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

# --- FastAPI App ---
app = FastAPI()

# Add the cybersecurity middleware
ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"
@app.middleware("http")
async def check_threat_ip_middleware(request: Request, call_next: Callable):
    client_ip = get_remote_address(request)
    
    if client_ip in ["127.0.0.1", "::1"]:
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

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    for course in courses:
        course_text = build_course_text(course)
        embedding = embedding_model.encode([course_text], show_progress_bar=False)[0]
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

    logger.info("[startup] Computing course embeddings...")
    cache_course_embeddings(courses)
    logger.info("[startup] Course embeddings cached successfully.")

try:
    # Load courses.json from the server folder (path relative to this file)
    courses_path = os.path.join(os.path.dirname(__file__), "courses.json")
    with open(courses_path, "r") as f:
        COURSES = json.load(f)
except Exception as e:
    # Log a friendly warning; downstream code expects COURSES to be a list.
    logger.warning(f"[startup] Warning: could not load courses.json: {e}")
    COURSES = []


def infer_course_difficulty(course: dict) -> str:
    """Simple heuristic to infer course difficulty from title or tags.
    Returns one of: 'easy', 'medium', 'hard'.
    """
    title = course.get("title", "").lower()
    tags = [t.lower() for t in course.get("tags", [])]

    hard_keywords = ["advanced", "expert", "deep learning", "machine learning algorithms", "reinforcement"]
    easy_keywords = ["basics", "fundamentals", "intro to", "introduction"]

    if any(keyword in title for keyword in hard_keywords) or any(k in " ".join(tags) for k in hard_keywords):
        return "hard"
    if any(keyword in title for keyword in easy_keywords) or any(k in " ".join(tags) for k in easy_keywords):
        return "easy"
    return "medium"

@app.get("/quiz", response_model=List[QuizQuestion])
async def get_quiz(
    difficulty: str = Query(...),
    topic: Optional[str] = Query(None)
):
    difficulty_mapping = {"easy": 1, "medium": 2, "hard": 3}
    if difficulty.lower() not in difficulty_mapping:
        raise HTTPException(status_code=400, detail="Invalid difficulty.")
    weight_value = difficulty_mapping[difficulty.lower()]
    match_query: Dict[str, Any] = {"weight": weight_value}
    if topic:
        match_query["topic"] = {"$regex": f".*{topic}.*", "$options": "i"}

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
        user_results_collection.update_one(
            {"user_id": result.user_id, "topic": result.topic},
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
        user_embedding = embedding_model.encode([user_text], normalize_embeddings=True)[0]

        # --- Retrieve course embeddings ---
        course_ids = [c["id"] for c in filtered_courses if c.get("id")]
        course_embeddings_docs = list(course_embeddings_collection.find({"courseId": {"$in": course_ids}}))
        logger.info(f"Retrieved {len(course_embeddings_docs)} course embeddings from DB.")

        emb_map = {doc["courseId"]: np.array(doc["embedding"]) for doc in course_embeddings_docs if "embedding" in doc}
        aligned_filtered_courses = [c for c in filtered_courses if c["id"] in emb_map]
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

@app.post("/chat/")
async def chat(input: ChatInput):
    try:
        logger.info(f"Processing chat request: message='{input.message}', user_id='{input.user_id}', conversation_id='{input.conversation_id}'")

        # Improved classification to recognize programming-related terms
        classification_prompt = PromptTemplate(
            input_variables=["message"],
            template="Classify the following user message as 'coding' or 'other'. Consider terms like 'Java', 'object', 'class', 'programming', 'code', or specific programming concepts as 'coding'. Message: {message} Classification:"
        )
        classification_chain = classification_prompt | llm | StrOutputParser()
        classification_result = await classification_chain.ainvoke({"message": input.message})
        logger.info(f"Message classification: {classification_result}")

        if "other" in classification_result.strip().lower():
            logger.info("Message classified as 'other', returning generic response")
            return {
                "response": "I specialize in computer science and engineering topics like Java, AI, and cybersecurity. For non-technical questions, please consult another resource.",
                "conversation_id": input.conversation_id
            }

        # Fetch user quiz results for context
        user_quiz_results = None
        if input.user_id:
            user_quiz_results = user_results_collection.find_one({"user_id": input.user_id})
            logger.info(f"User quiz results: {user_quiz_results}")

        # Retrieve relevant quiz questions, prioritizing topic matches
        query_terms = input.message.lower().split()
        topic_regex = "|".join([f".*{term}.*" for term in query_terms if term in ["java", "object", "class", "programming"]])
        retrieved_questions = list(quiz_collection.find({
            "$or": [
                {"question": {"$regex": input.message, "$options": "i"}},
                {"topic": {"$regex": topic_regex, "$options": "i"}}
            ]
        }).limit(5))
        logger.info(f"Retrieved {len(retrieved_questions)} quiz questions for context")

        # Build context
        context = ""
        if user_quiz_results:
            context += f"User has a past quiz score of {user_quiz_results.get('score', 'N/A')}% on topic '{user_quiz_results.get('topic', 'N/A')}'.\n\n"
        if retrieved_questions:
            context += "Relevant quiz questions for context:\n"
            for q in retrieved_questions:
                correct_option_index = q.get('correctAnswer')
                correct_answer = q['options'][correct_option_index] if 'options' in q and correct_option_index is not None else "N/A"
                context += f"- Question: {q.get('question', 'N/A')}\n  Correct Answer: {correct_answer}\n"
        else:
            context += "No specific quiz questions found for this topic.\n"

        # Enhanced prompt for technical questions
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="You are a knowledgeable AI assistant specializing in computer science. Provide a clear, accurate, and concise answer to the user's question, using the provided context if relevant. For Java-related questions, explain concepts in the context of Java programming, including examples where appropriate. Context: {context}\nUser's Question: {question}\nAnswer:"
        )
        chain = prompt_template | llm | StrOutputParser()
        response = await chain.ainvoke({"context": context, "question": input.message})
        logger.info(f"Chat response generated: {response[:100]}...")

        return {"response": response, "conversation_id": input.conversation_id}

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat request: {str(e)}")

# Serve static files from client-test folder
app.mount("/static", StaticFiles(directory="../client-test"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("../client-test/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
