import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { BarChart3, Trophy, Flame, BookOpen, Star, Calendar, TrendingUp } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { apiHelpers } from '../config/api';
import StarBackground from '../components/UI/StarBackground';
import LoadingSpinner from '../components/UI/LoadingSpinner';

interface QuizResult {
  id: string;
  topic: string;
  profession: string;
  score: number;
  difficulty: string;
  date: string;
}

interface Course {
  title: string;
  description: string;
  tags: string[];
  relevance: number;
}

const Dashboard: React.FC = () => {
  const { user } = useAuth();
  const [quizResults, setQuizResults] = useState<QuizResult[]>([]);
  const [recommendations, setRecommendations] = useState<Course[]>([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalQuizzes: 0,
    averageScore: 0,
    currentStreak: 7, // Mock data - would come from backend
    totalTopics: 0
  });

  useEffect(() => {
    if (user) {
      loadDashboardData();
    }
  }, [user]);

  const loadDashboardData = async () => {
    if (!user) return;

    try {
      // Load quiz results
      const results = await apiHelpers.getUserQuizResults(user.uid);
      setQuizResults(results);

      // Calculate stats
      const totalQuizzes = results.length;
      const averageScore = results.length > 0 
        ? Math.round(results.reduce((sum: number, result: QuizResult) => sum + result.score, 0) / results.length)
        : 0;
      const totalTopics = new Set(results.map((result: QuizResult) => result.topic)).size;

      setStats({
        totalQuizzes,
        averageScore,
        currentStreak: 7, // Mock data
        totalTopics
      });

      // Load recommendations based on most recent profession/topics
      if (results.length > 0) {
        const recentResult = results[results.length - 1];
        const recs = await apiHelpers.getRecommendations(
          user.uid,
          recentResult.profession,
          { topic: recentResult.topic }
        );
        setRecommendations(recs.slice(0, 3)); // Show top 3
      }
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-400';
    if (score >= 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getScoreBg = (score: number) => {
    if (score >= 80) return 'bg-green-400/20 border-green-400/30';
    if (score >= 60) return 'bg-yellow-400/20 border-yellow-400/30';
    return 'bg-red-400/20 border-red-400/30';
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
      
      <div className="container mx-auto px-6 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            Your Learning Dashboard
          </h1>
          <p className="text-gray-300 text-lg">
            Welcome back, {user?.email}! Here's your cosmic learning journey.
          </p>
        </motion.div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-purple-500/20"
          >
            <div className="flex items-center space-x-3">
              <Trophy className="h-8 w-8 text-yellow-400" />
              <div>
                <div className="text-2xl font-bold text-white">{stats.totalQuizzes}</div>
                <div className="text-gray-400">Total Quizzes</div>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-green-500/20"
          >
            <div className="flex items-center space-x-3">
              <BarChart3 className="h-8 w-8 text-green-400" />
              <div>
                <div className="text-2xl font-bold text-white">{stats.averageScore}%</div>
                <div className="text-gray-400">Average Score</div>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-red-500/20"
          >
            <div className="flex items-center space-x-3">
              <Flame className="h-8 w-8 text-red-400" />
              <div>
                <div className="text-2xl font-bold text-white">{stats.currentStreak}</div>
                <div className="text-gray-400">Day Streak</div>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-cyan-500/20"
          >
            <div className="flex items-center space-x-3">
              <BookOpen className="h-8 w-8 text-cyan-400" />
              <div>
                <div className="text-2xl font-bold text-white">{stats.totalTopics}</div>
                <div className="text-gray-400">Topics Explored</div>
              </div>
            </div>
          </motion.div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Recent Quiz Results */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
            className="lg:col-span-2"
          >
            <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              Recent Quiz Results
            </h2>
            
            {quizResults.length === 0 ? (
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-8 border border-purple-500/20 text-center">
                <Trophy className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-400 text-lg">No quiz results yet!</p>
                <p className="text-gray-500 mb-4">Take your first quiz to see your progress here.</p>
                <button
                  onClick={() => window.location.href = '/quiz'}
                  className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 rounded-lg font-semibold transform hover:scale-105 transition-all"
                >
                  Take a Quiz
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                {quizResults.slice(0, 5).map((result, index) => (
                  <motion.div
                    key={result.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6 + index * 0.1 }}
                    className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-purple-500/20 hover:border-purple-500/40 transition-all"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-lg font-semibold text-white">{result.topic}</h3>
                      <div className={`px-3 py-1 rounded-full border ${getScoreBg(result.score)}`}>
                        <span className={`font-bold ${getScoreColor(result.score)}`}>
                          {result.score}%
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-sm text-gray-400">
                      <span>{result.profession}</span>
                      <div className="flex items-center space-x-3">
                        <span className="capitalize">{result.difficulty}</span>
                        <span className="flex items-center space-x-1">
                          <Calendar className="h-4 w-4" />
                          <span>{new Date(result.date).toLocaleDateString()}</span>
                        </span>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </motion.div>

          {/* Recommendations */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.7 }}
          >
            <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
              Recommended for You
            </h2>
            
            {recommendations.length === 0 ? (
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-cyan-500/20 text-center">
                <TrendingUp className="h-8 w-8 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-400">Take a quiz to get personalized recommendations!</p>
              </div>
            ) : (
              <div className="space-y-4">
                {recommendations.map((course, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.8 + index * 0.1 }}
                    className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-cyan-500/20 hover:border-cyan-500/40 transition-all"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <h3 className="text-lg font-semibold text-white leading-tight">
                        {course.title}
                      </h3>
                      <div className="flex items-center space-x-1 ml-2">
                        <Star className="h-4 w-4 text-yellow-400" />
                        <span className="text-yellow-400 text-sm font-semibold">
                          {course.relevance}%
                        </span>
                      </div>
                    </div>
                    <p className="text-gray-300 text-sm mb-3 line-clamp-2">
                      {course.description}
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {course.tags.slice(0, 2).map((tag, tagIndex) => (
                        <span
                          key={tagIndex}
                          className="px-2 py-1 bg-cyan-600/20 text-cyan-300 text-xs rounded-full"
                        >
                          {tag}
                        </span>
                      ))}
                      {course.tags.length > 2 && (
                        <span className="px-2 py-1 bg-slate-600/30 text-gray-400 text-xs rounded-full">
                          +{course.tags.length - 2} more
                        </span>
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;