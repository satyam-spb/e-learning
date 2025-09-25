import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// API helper functions
export const apiHelpers = {
  async getQuiz(difficulty: string, topic: string) {
    const response = await api.get(
      `/quiz?difficulty=${difficulty}&topic=${topic}`
    );
    return response.data;
  },

  async submitQuizResult(
    userId: string | null,
    topic: string,
    profession: string,
    score: number,
    difficulty: string
  ) {
    const response = await api.post("/quiz/results", {
      // backend expects a string for user_id; send empty string when no user
      user_id: userId ?? "",
      topic,
      profession,
      score,
      difficulty,
    });
    return response.data;
  },

  async getRecommendations(
    userId: string | null,
    profession: string,
    preferences: any = {}
  ) {
    // The backend expects preferences as a list of strings.
    // If callers pass an object (e.g. { topic, difficulty }), convert it to an array.
    let prefsArray: string[] = [];
    if (Array.isArray(preferences)) {
      prefsArray = preferences;
    } else if (preferences && typeof preferences === "object") {
      // include topic and difficulty if present
      if (preferences.topic) prefsArray.push(String(preferences.topic));
      if (preferences.difficulty)
        prefsArray.push(String(preferences.difficulty));
    }

    const response = await api.post("/recommend", {
      // backend expects a string for user_id; send empty string when no user
      user_id: userId ?? "",
      profession,
      preferences: prefsArray,
    });

    // Backend returns { recommendations: [...] } — return the array for consumers.
    return response.data?.recommendations ?? [];
  },

  async sendChatMessage(
    message: string,
    conversationId: string,
    userId: string | null
  ) {
    const response = await api.post("/chat/", {
      message,
      conversation_id: conversationId,
      // backend expects a string for user_id; send empty when no user
      user_id: userId ?? "",
    });
    return response.data;
  },

  async getUserQuizResults(userId: string) {
    const response = await api.get(`/quiz/results?user_id=${userId}`);
    return response.data;
  },
};
