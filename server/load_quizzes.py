import os
import json
import pymongo
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
    raise SystemExit("MONGO_URI not set. Export it in your shell or provide an ENV_FILE_PATH pointing to a file that contains it for production.")
DB_NAME = "elearning-ai"
QUIZ_COLLECTION = "quizzes"

def load_data():
    client = None
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        quizzes_coll = db[QUIZ_COLLECTION]

        quizzes_coll.delete_many({})
        
        with open(os.path.join(os.path.dirname(__file__), "quiz.json"), "r") as f:
            topic_data = json.load(f)

        all_questions = []
        for topic, questions in topic_data.items():
            if isinstance(questions, list):
                for q in questions:
                    q['topic'] = topic
                    all_questions.append(q)

        if all_questions:
            quizzes_coll.insert_many(all_questions)
        
        print(f"Successfully loaded {len(all_questions)} quizzes into MongoDB.")

    except FileNotFoundError:
        print("Error: quiz.json not found. Please ensure the file is in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if client:
            client.close()

if __name__ == "__main__":
    load_data()