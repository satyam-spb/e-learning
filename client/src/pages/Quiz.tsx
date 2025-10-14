import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, ChevronLeft, ChevronRight, CheckCircle } from "lucide-react";
import { useQuiz } from "../context/QuizContext";
import { useAuth } from "../context/AuthContext";
import { apiHelpers } from "../config/api";
import { toast } from "react-toastify";
import LoadingSpinner from "../components/UI/LoadingSpinner";
import StarBackground from "../components/UI/StarBackground";

const Quiz: React.FC = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const {
    quizState,
    setQuizConfig,
    setQuestions,
    answerQuestion,
    nextQuestion,
    previousQuestion,
    completeQuiz,
  } = useQuiz();

  const [showConfig, setShowConfig] = useState(true);
  const [loading, setLoading] = useState(false);
  const [configForm, setConfigForm] = useState({
    profession: "",
    topic: "",
    difficulty: "medium",
  });
  const [professionOptions, setProfessionOptions] = useState<string[]>([]);
  // Restrict topics to a curated allowed list to match quiz coverage
  const allowedTopics = [
    "AI Model Deployment & MLOps",
    "APIs and RESTful Services",
    "Agile & DevOps Practices",
    "Android Development",
    "Backend Development (Node.js, Express)",
    "C/C++",
    "Clean Code & Code Refactoring",
    "Cloud Computing (AWS, Azure, GCP)",
    "Data Science with Python/R",
    "Data Structures and Algorithms",
    "Deep Learning & Neural Networks",
    "Design Patterns & Architecture",
    "Docker & Containerization",
    "Frontend Development (HTML, CSS, JS)",
    "Full-Stack Web Development",
    "Functional Programming",
    "GraphQL Fundamentals",
    "Infrastructure as Code (Terraform)",
    "JavaScript",
    "Kubernetes & Microservices",
    "Machine Learning Basics",
    "Memory Management & Pointers",
    "Mobile App Deployment",
    "Natural Language Processing (NLP)",
    "Nextjs",
    "ORMs and Database Optimization",
    "Object-Oriented Programming (OOP)",
    "Progressive Web Apps",
    "Python",
    "React Native / Flutter",
    "Recursion and Dynamic Programming",
    "SQL & Relational Databases",
    "Serverless Architecture",
    "Testing & Debugging",
    "TypeScript",
    "Version Control (Git & GitHub)",
    "ios development",
    "java",
    "noSQL",
    "react",
  ];

  const [topicOptions, setTopicOptions] = useState<string[]>(allowedTopics);

  const currentQuestion = quizState.questions[quizState.currentQuestionIndex];
  const progress =
    quizState.questions.length > 0
      ? ((quizState.currentQuestionIndex + 1) / quizState.questions.length) *
        100
      : 0;

  useEffect(() => {
    // If we have an ongoing quiz, don't show config
    if (quizState.questions.length > 0 && !quizState.isCompleted) {
      setShowConfig(false);
    }
  }, [quizState]);

  useEffect(() => {
    // load courses to populate profession/topic selects
    const loadOptions = async () => {
      try {
        const courses = await apiHelpers.getCourses();
        const profSet = new Set<string>();
        courses.forEach((c: any) => {
          if (Array.isArray(c.profession))
            c.profession.forEach((p: string) => profSet.add(p));
          // we do not populate topics from course tags; topics are curated
        });
        setProfessionOptions(Array.from(profSet).sort());
        // topics remain the curated allowed list
        setTopicOptions(allowedTopics);
      } catch (err) {
        console.error("Failed to load course options:", err);
      }
    };
    loadOptions();
  }, []);

  const handleStartQuiz = async () => {
    if (!configForm.profession || !configForm.topic) {
      toast.error("Please fill in all fields");
      return;
    }

    setLoading(true);
    try {
      const questions = await apiHelpers.getQuiz(
        configForm.difficulty,
        configForm.topic
      );
      setQuizConfig(
        configForm.profession,
        configForm.topic,
        configForm.difficulty
      );
      setQuestions(questions);
      setShowConfig(false);
      toast.success("The spotlight’s on you—let’s see how you shine!");
    } catch (error) {
      toast.error(
        "We're still building this quiz. You might discover more by tweaking your topic or profession slightly."
      );
      console.error("Failed to load quiz:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleAnswerSelect = (answer: string) => {
    answerQuestion(currentQuestion.id, answer);
  };

  const handleNext = () => {
    if (quizState.currentQuestionIndex === quizState.questions.length - 1) {
      // Last question - complete quiz
      handleCompleteQuiz();
    } else {
      nextQuestion();
    }
  };

  const handleCompleteQuiz = async () => {
    completeQuiz();

    // Save quiz result to backend
    try {
      await apiHelpers.submitQuizResult(
        user?.uid || null,
        quizState.topic,
        quizState.profession,
        quizState.score,
        quizState.difficulty
      );
    } catch (error) {
      console.error("Failed to save quiz result:", error);
      // Cache the result locally if user is not logged in
      const cachedResults = JSON.parse(
        localStorage.getItem("cachedQuizResults") || "[]"
      );
      cachedResults.push({
        topic: quizState.topic,
        profession: quizState.profession,
        score: quizState.score,
        difficulty: quizState.difficulty,
        date: new Date().toISOString(),
      });
      localStorage.setItem("cachedQuizResults", JSON.stringify(cachedResults));
    }

    navigate("/quiz/result");
  };

  if (showConfig) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white relative overflow-hidden">
        <StarBackground />

        <div className="container mx-auto px-6 py-20">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="max-w-2xl mx-auto"
          >
            <div className="text-center mb-8">
              <Brain className="h-16 w-16 text-purple-400 mx-auto mb-4" />
              <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Configure Your Quiz
              </h1>
              <p className="text-gray-300 text-lg">
                Let's create a personalized quiz based on your profession and
                interests
              </p>
            </div>

            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-8 border border-purple-500/20">
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    What's your profession?
                  </label>
                  <input
                    list="profession-list"
                    value={configForm.profession}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                      setConfigForm((prev) => ({
                        ...prev,
                        profession: e.target.value,
                      }))
                    }
                    placeholder="Type or select profession"
                    className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg focus:outline-none focus:border-purple-400 text-white"
                    aria-label="Profession"
                  />
                  <datalist id="profession-list">
                    {professionOptions.map((p) => (
                      <option key={p} value={p} />
                    ))}
                  </datalist>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    What topic would you like to be quizzed on?
                  </label>
                  <input
                    list="topic-list"
                    value={configForm.topic}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                      setConfigForm((prev) => ({
                        ...prev,
                        topic: e.target.value,
                      }))
                    }
                    placeholder="Type or select topic"
                    className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg focus:outline-none focus:border-purple-400 text-white"
                    aria-label="Topic"
                  />
                  <datalist id="topic-list">
                    {topicOptions.map((t) => (
                      <option key={t} value={t} />
                    ))}
                  </datalist>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Choose difficulty level
                  </label>
                  <select
                    value={configForm.difficulty}
                    onChange={(e) =>
                      setConfigForm((prev) => ({
                        ...prev,
                        difficulty: e.target.value,
                      }))
                    }
                    className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-lg focus:outline-none focus:border-purple-400 text-white"
                  >
                    <option value="easy">Easy</option>
                    <option value="medium">Medium</option>
                    <option value="hard">Hard</option>
                  </select>
                </div>

                <button
                  onClick={handleStartQuiz}
                  disabled={loading}
                  className="w-full px-6 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 rounded-lg font-semibold transform hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                >
                  {loading ? (
                    <LoadingSpinner />
                  ) : (
                    <>
                      <Brain className="h-5 w-5" />
                      <span>Start Quiz</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    );
  }

  if (!currentQuestion) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white flex items-center justify-center">
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white relative overflow-hidden">
      <StarBackground />

      <div className="container mx-auto px-6 py-8">
        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex items-center justify-between text-sm text-gray-400 mb-2">
            <span>
              Question {quizState.currentQuestionIndex + 1} of{" "}
              {quizState.questions.length}
            </span>
            <span>{Math.round(progress)}% Complete</span>
          </div>
          <div className="w-full bg-slate-700 rounded-full h-2">
            <motion.div
              className="bg-gradient-to-r from-purple-600 to-pink-600 h-2 rounded-full"
              style={{ width: `${progress}%` }}
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>

        {/* Quiz Content */}
        <div className="max-w-4xl mx-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={quizState.currentQuestionIndex}
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              transition={{ duration: 0.3 }}
              className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-8 border border-purple-500/20 mb-8"
            >
              <h2 className="text-2xl font-bold mb-6 text-white">
                {currentQuestion.question}
              </h2>

              <div className="space-y-3">
                {currentQuestion.options.map((option, index) => {
                  const isSelected =
                    quizState.answers[currentQuestion.id] === option;
                  return (
                    <motion.button
                      key={index}
                      onClick={() => handleAnswerSelect(option)}
                      className={`w-full p-4 text-left rounded-lg border-2 transition-all ${
                        isSelected
                          ? "border-purple-400 bg-purple-600/20"
                          : "border-slate-600 bg-slate-700/30 hover:border-slate-500 hover:bg-slate-600/30"
                      }`}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="flex items-center space-x-3">
                        <div
                          className={`w-6 h-6 rounded-full border-2 flex items-center justify-center ${
                            isSelected
                              ? "border-purple-400 bg-purple-400"
                              : "border-gray-500"
                          }`}
                        >
                          {isSelected && (
                            <CheckCircle className="h-4 w-4 text-white" />
                          )}
                        </div>
                        <span className="text-white">{option}</span>
                      </div>
                    </motion.button>
                  );
                })}
              </div>
            </motion.div>
          </AnimatePresence>

          {/* Navigation */}
          <div className="flex items-center justify-between">
            <button
              onClick={previousQuestion}
              disabled={quizState.currentQuestionIndex === 0}
              className="flex items-center space-x-2 px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeft className="h-5 w-5" />
              <span>Previous</span>
            </button>

            <div className="flex items-center space-x-4">
              <span className="text-gray-400">
                {Object.keys(quizState.answers).length} /{" "}
                {quizState.questions.length} answered
              </span>
            </div>

            <button
              onClick={handleNext}
              disabled={!quizState.answers[currentQuestion.id]}
              className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <span>
                {quizState.currentQuestionIndex ===
                quizState.questions.length - 1
                  ? "Complete"
                  : "Next"}
              </span>
              {quizState.currentQuestionIndex !==
                quizState.questions.length - 1 && (
                <ChevronRight className="h-5 w-5" />
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Quiz;
