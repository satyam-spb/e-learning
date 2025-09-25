import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Trophy, Star, BookOpen, MessageCircle, Home, RefreshCw } from 'lucide-react';
import { useQuiz } from '../context/QuizContext';
import { useAuth } from '../context/AuthContext';
import { apiHelpers } from '../config/api';
import StarBackground from '../components/UI/StarBackground';
import LoadingSpinner from '../components/UI/LoadingSpinner';

interface Course {
  title: string;
  description: string;
  tags: string[];
  relevance: number;
}

const QuizResult: React.FC = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { quizState, resetQuiz } = useQuiz();
  const [recommendations, setRecommendations] = useState<Course[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!quizState.isCompleted) {
      navigate('/quiz');
      return;
    }

    loadRecommendations();
  }, [quizState.isCompleted, navigate]);

  const loadRecommendations = async () => {
    try {
      const recs = await apiHelpers.getRecommendations(
        user?.uid || null,
        quizState.profession,
        { topic: quizState.topic, difficulty: quizState.difficulty }
      );
      setRecommendations(recs);
    } catch (error) {
      console.error('Failed to load recommendations:', error);
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'from-green-400 to-green-600';
    if (score >= 60) return 'from-yellow-400 to-yellow-600';
    return 'from-red-400 to-red-600';
  };

  const getScoreMessage = (score: number) => {
    if (score >= 90) return 'Outstanding! You\'re a cosmic champion! 🌟';
    if (score >= 80) return 'Excellent work! You\'re among the stars! ✨';
    if (score >= 70) return 'Great job! You\'re on the right trajectory! 🚀';
    if (score >= 60) return 'Good effort! Keep exploring the cosmos! 🌍';
    return 'Keep learning! Every star started as stardust! 💫';
  };

  const handleRetakeQuiz = () => {
    resetQuiz();
    navigate('/quiz');
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white flex items-center justify-center">
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white relative overflow-hidden">
      <StarBackground />
      
      <div className="container mx-auto px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="max-w-4xl mx-auto"
        >
          {/* Score Card */}
          <div className="text-center mb-12">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="mb-6"
            >
              <Trophy className="h-20 w-20 text-yellow-400 mx-auto mb-4" />
              <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Quiz Complete!
              </h1>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              className={`inline-block px-8 py-6 bg-gradient-to-r ${getScoreColor(quizState.score)} rounded-full text-white mb-4`}
            >
              <div className="text-6xl font-bold">{quizState.score}%</div>
              <div className="text-lg">Your Score</div>
            </motion.div>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.6 }}
              className="text-xl text-gray-300 mb-8"
            >
              {getScoreMessage(quizState.score)}
            </motion.p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.7 }}
                className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-4 border border-purple-500/20"
              >
                <div className="text-2xl font-bold text-purple-400">{quizState.topic}</div>
                <div className="text-gray-400">Topic</div>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.8 }}
                className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-4 border border-purple-500/20"
              >
                <div className="text-2xl font-bold text-cyan-400 capitalize">{quizState.difficulty}</div>
                <div className="text-gray-400">Difficulty</div>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.9 }}
                className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-4 border border-purple-500/20"
              >
                <div className="text-2xl font-bold text-pink-400">{quizState.questions.length}</div>
                <div className="text-gray-400">Questions</div>
              </motion.div>
            </div>
          </div>

          {/* Recommendations Section */}
          {recommendations.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 1 }}
              className="mb-12"
            >
              <h2 className="text-3xl font-bold mb-8 text-center bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Recommended Courses
              </h2>
              <div className="grid gap-6">
                {recommendations.map((course, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: 1.2 + index * 0.1 }}
                    className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-purple-500/20 hover:border-purple-500/40 transition-all"
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <h3 className="text-xl font-bold text-white mb-2">{course.title}</h3>
                        <p className="text-gray-300 mb-4">{course.description}</p>
                      </div>
                      <div className="flex items-center space-x-1 ml-4">
                        <Star className="h-5 w-5 text-yellow-400" />
                        <span className="text-yellow-400 font-semibold">{course.relevance}%</span>
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {course.tags.map((tag, tagIndex) => (
                        <span
                          key={tagIndex}
                          className="px-3 py-1 bg-purple-600/30 text-purple-300 text-sm rounded-full"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Action Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1.5 }}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <button
              onClick={handleRetakeQuiz}
              className="flex items-center justify-center space-x-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 rounded-lg font-semibold transform hover:scale-105 transition-all"
            >
              <RefreshCw className="h-5 w-5" />
              <span>Retake Quiz</span>
            </button>

            <Link
              to="/chatbot"
              className="flex items-center justify-center space-x-2 px-6 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 rounded-lg font-semibold transform hover:scale-105 transition-all"
            >
              <MessageCircle className="h-5 w-5" />
              <span>Chat with AI</span>
            </Link>

            <Link
              to="/"
              className="flex items-center justify-center space-x-2 px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg font-semibold transform hover:scale-105 transition-all"
            >
              <Home className="h-5 w-5" />
              <span>Go Home</span>
            </Link>
          </motion.div>

          {/* Sign Up Prompt for Non-users */}
          {!user && (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 1.7 }}
              className="mt-8 p-6 bg-gradient-to-r from-purple-800/50 to-pink-800/50 rounded-xl border border-purple-500/30 text-center"
            >
              <BookOpen className="h-8 w-8 text-purple-400 mx-auto mb-4" />
              <h3 className="text-xl font-bold mb-2">Save Your Progress</h3>
              <p className="text-gray-300 mb-4">
                Sign up to track your quiz history, maintain learning streaks, and get personalized recommendations!
              </p>
              <Link
                to="/auth"
                className="inline-flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-green-600 to-teal-600 hover:from-green-700 hover:to-teal-700 rounded-lg font-semibold transform hover:scale-105 transition-all"
              >
                <span>Sign Up Now</span>
              </Link>
            </motion.div>
          )}
        </motion.div>
      </div>
    </div>
  );
};

export default QuizResult;