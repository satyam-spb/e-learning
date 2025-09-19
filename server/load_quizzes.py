import os
import json
import pymongo
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI") or "mongodb://localhost:27017/"
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