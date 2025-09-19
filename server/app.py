import os
import json
from typing import List, Dict, Optional, Callable
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import random
import requests
from slowapi.util import get_remote_address

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

# Load embedding model once for RAG and recommendations
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Setup Groq client
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)

# --- FastAPI App ---
app = FastAPI()

# Add the cybersecurity middleware
ABUSEIPDB_URL = "https://api.abuseipdb.com/api/v2/check"
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
        return await call_next(request)

app.middleware("http")(check_threat_ip_middleware)

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
    user_id: str

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

# --- Global Data ---
try:
    with open("courses.json", "r") as f:
        COURSES = json.load(f)
except FileNotFoundError:
    COURSES = []

# --- Helper Functions ---
def build_course_text(course: dict) -> str:
    title = course.get("title", "")
    desc = course.get("description", "")
    tags = ", ".join(course.get("tags", []))
    profession = ", ".join(course.get("profession", []))
    return f"Title: {title}, Description: {desc}, Tags: {tags}, Profession: {profession}"

def cache_course_embeddings(courses: List[dict]):
    for course in courses:
        course_text = build_course_text(course)
        embedding = embedding_model.encode([course_text], show_progress_bar=False)[0]
        course_id = course["id"]
        course_embeddings_collection.update_one(
            {"courseId": course_id},
            {"$set": {"embedding": embedding.tolist()}},
            upsert=True
        )

# --- Startup: cache embeddings ---
@app.on_event("startup")
async def startup_event():
    if COURSES:
        cache_course_embeddings(COURSES)

def infer_course_difficulty(course: dict) -> str:
    title = course.get("title", "").lower()
    tags = [t.lower() for t in course.get("tags", [])]
    
    if any(keyword in title for keyword in ["advanced", "expert", "deep learning", "machine learning algorithms", "reinforcement"]):
        return "hard"
    if any(keyword in title for keyword in ["basics", "fundamentals", "intro to", "introduction"]):
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
    match_query = {"weight": weight_value}
    if topic:
        match_query["topic"] = {"$regex": f"^{topic}$", "$options": "i"}

    try:
        questions = list(quiz_collection.aggregate([
            {"$match": match_query},
            {"$sample": {"size": 10}}
        ]))
        if not questions:
            raise HTTPException(status_code=404, detail=f"No questions found.")
        return questions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/quiz/results")
async def save_quiz_results(result: UserQuizResult):
    try:
        user_results_collection.update_one(
            {"user_id": result.user_id, "topic": result.topic},
            {"$set": result.dict()},
            upsert=True
        )
        return {"message": "Quiz results saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving quiz results: {str(e)}")

@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    try:
        user_quiz = None
        user_quiz_result = user_results_collection.find_one({"user_id": request.user_id})
        if user_quiz_result:
            user_quiz = Quiz(score=user_quiz_result["score"], course=user_quiz_result["topic"])
        
        user_prefs = set(p.strip().lower() for p in request.preferences)
        strict_courses = [
            c for c in COURSES
            if (user_quiz and user_quiz.course.lower() in (c["title"].lower() + " " + c["description"].lower())) or
               any(p in c["tags"] for p in user_prefs) or
               (request.profession in c["profession"])
        ]
        if user_quiz and user_quiz.course.lower() == "java":
            filtered_courses = [c for c in strict_courses if "javascript" not in c["tags"]]
        else:
            filtered_courses = strict_courses or COURSES
        
        quiz_part = f"Quiz: {user_quiz.course} with score {user_quiz.score}%" if user_quiz else "No quiz"
        user_text = f"Profession: {request.profession}, Preferences: {', '.join(request.preferences)}, {quiz_part}"
        user_embedding = embedding_model.encode([user_text])[0]
        
        course_embeddings = [np.array(c["embedding"]) for c in course_embeddings_collection.find({"courseId": {"$in": [c["id"] for c in filtered_courses]}})]
        similarities = cosine_similarity([user_embedding], course_embeddings)[0]
        
        user_proficiency = "none"
        if user_quiz:
            user_proficiency = "high" if user_quiz.score >= 80 else "medium" if user_quiz.score >= 50 else "low"
        
        relevance_scores = []
        for i, course in enumerate(filtered_courses):
            semantic_score = similarities[i]
            course_difficulty = infer_course_difficulty(course)
            difficulty_multiplier = 1.0
            if user_proficiency == "high" and course_difficulty in ["easy", "medium"]: difficulty_multiplier = 1.1
            elif user_proficiency == "medium" and course_difficulty == "medium": difficulty_multiplier = 1.2
            elif user_proficiency == "low" and course_difficulty == "easy": difficulty_multiplier = 1.3
            elif user_proficiency == "high" and course_difficulty == "hard": difficulty_multiplier = 0.8
            elif user_proficiency == "low" and course_difficulty == "hard": difficulty_multiplier = 0.5
            final_relevance = min(1.0, semantic_score * difficulty_multiplier) * 100
            relevance_scores.append(final_relevance)
        
        sorted_indices = np.argsort(relevance_scores)[::-1]
        top_courses_with_relevance = []
        for i in sorted_indices[:10]:
            course = filtered_courses[i]
            course_with_relevance = course.copy()
            course_with_relevance["relevance"] = int(relevance_scores[i])
            top_courses_with_relevance.append(course_with_relevance)
        return {"recommendations": top_courses_with_relevance}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/chat/")
async def chat(input: ChatInput):
    try:
        classification_prompt = PromptTemplate(input_variables=["message"], template="Classify the following user message as 'coding' or 'other'. Message: {message} Classification:")
        classification_chain = classification_prompt | llm | StrOutputParser()
        classification_result = await classification_chain.ainvoke({"message": input.message})
        
        if "other" in classification_result.strip().lower():
            return {"response": "I am a chatbot specializing in computer science and engineering. I can guide you on topics related to AI, software development, and cybersecurity. For other fields, I recommend seeking out a different resource.", "conversation_id": input.conversation_id}

        user_quiz_results = user_results_collection.find_one({"user_id": input.user_id})
        retrieved_questions = list(quiz_collection.find({"question": {"$regex": input.message, "$options": "i"}}).limit(5))
        context = ""
        if user_quiz_results:
            context += f"The user has a past quiz score of {user_quiz_results.get('score', 'N/A')}% on the topic '{user_quiz_results.get('topic', 'N/A')}'.\n\n"
        if retrieved_questions:
            context += "Relevant quiz questions for context:\n"
            for q in retrieved_questions:
                correct_option_index = q.get('correctAnswer')
                correct_answer = q['options'][correct_option_index] if 'options' in q and correct_option_index is not None else "N/A"
                context += f"- Question: {q.get('question', 'N/A')}\n- Correct Answer: {correct_answer}\n"

        prompt_template = PromptTemplate(input_variables=["context", "question"], template="You are a helpful and professional AI assistant. Use the following context to inform your response. Context: {context} User's Question: {question}")
        chain = prompt_template | llm | StrOutputParser()
        response = await chain.ainvoke({"context": context, "question": input.message})
        return {"response": response, "conversation_id": input.conversation_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI assistant is unavailable. Please try again later. (Error: {str(e)})")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)