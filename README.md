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
- **Frontend:** HTML, CSS, JavaScript (for testing purposes)

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
├── load_quizzes.py <-- Script for seeding the database
├── courses.json <-- Course data file
├── quiz.json <-- Quiz question data file
└── requirements.txt <-- All Python dependencies

---

## ⚙️ Setup and Installation

Follow these steps to get the application running on your local machine.

### Step 1: Install MongoDB

This application requires a running MongoDB instance.

- Download and install MongoDB Community Server from the [official website](https://www.mongodb.com/try/download/community).
- Ensure the MongoDB service is running in the background.

### Step 2: Set Up Python and Virtual Environment

1.  Navigate to your project's `unified-ai-service` directory in your terminal.
2.  Create a virtual environment:
    ```bash
    python -m venv venv
    ```
3.  Activate the virtual environment:
    - **Windows (PowerShell):** `./venv/Scripts/Activate.ps1`
    - **macOS/Linux:** `source venv/bin/activate`

### Step 3: Install Required Libraries

With the virtual environment active, install all the Python dependencies from the `requirements.txt` file.

```bash
pip install -r requirements.txt
Step 4: Configure Environment Variables
Create a file named .env in the root directory of your project (e-learning).

GROQ_API_KEY="your_groq_api_key_here"
ABUSEIPDB_API_KEY="your_abuseipdb_api_key_here"
MONGO_URI="mongodb://localhost:27017/"
▶️ How to Run the Code
1. Seed the Database
You only need to run this script once to load your quiz questions from quiz.json into MongoDB.

Bash

python load_quizzes.py
2. Start the Backend Server
This one command starts the entire application. It runs the FastAPI server, which hosts all the application's endpoints.

Bash

uvicorn app:app --reload
The server will now be running on http://127.0.0.1:8000.

🧪 Testing the Application
Once the server is running, you can open the index.html file in your browser to test the full functionality.
```
