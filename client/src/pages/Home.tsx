import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Brain, MessageCircle, BarChart3, Sparkles, Zap, Target } from 'lucide-react';
import StarBackground from '../components/UI/StarBackground';

const Home: React.FC = () => {
  const scrollToSection = (elementId: string) => {
    document.getElementById(elementId)?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white relative overflow-hidden">
      <StarBackground />
      
      {/* Hero Section */}
      <section className="relative pt-20 pb-32 px-6">
        <div className="container mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-8"
          >
            <h1 className="text-5xl md:text-7xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent mb-6">
              Learn Beyond Limits
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto">
              Experience AI-powered learning that adapts to your profession and goals. 
              Take quizzes, chat with our cosmic AI, and track your progress across the universe of knowledge.
            </p>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <Link
              to="/quiz"
              className="px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 rounded-lg text-lg font-semibold transform hover:scale-105 transition-all flex items-center justify-center space-x-2"
            >
              <Brain className="h-5 w-5" />
              <span>Take a Quiz</span>
            </Link>
            <Link
              to="/chatbot"
              className="px-8 py-4 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 rounded-lg text-lg font-semibold transform hover:scale-105 transition-all flex items-center justify-center space-x-2"
            >
              <MessageCircle className="h-5 w-5" />
              <span>Chat with AI</span>
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-6 relative">
        <div className="container mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold mb-6 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              Cosmic Features
            </h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Discover the power of AI-driven personalized learning <br />(Scroll down to explore our key features.)
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.1 }}
              className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-8 border border-purple-500/20 hover:border-purple-500/40 transition-all"
            >
              <Brain className="h-12 w-12 text-purple-400 mb-6" />
              <h3 className="text-2xl font-bold mb-4">Adaptive Quizzes</h3>
              <p className="text-gray-300 mb-6">
                Personalized quizzes based on your profession and skill level. 
                Get instant feedback and tailored recommendations.
              </p>
              <button
                onClick={() => scrollToSection('quiz-section')}
                className="text-purple-400 hover:text-purple-300 font-semibold"
              >
                Learn More →
              </button>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-8 border border-cyan-500/20 hover:border-cyan-500/40 transition-all"
            >
              <MessageCircle className="h-12 w-12 text-cyan-400 mb-6" />
              <h3 className="text-2xl font-bold mb-4">Cosmic AI Chat</h3>
              <p className="text-gray-300 mb-6">
                Chat with our AI assistant in a beautiful cosmic environment. 
                Get answers, explanations, and learning support 24/7.
              </p>
              <button
                onClick={() => scrollToSection('chat-section')}
                className="text-cyan-400 hover:text-cyan-300 font-semibold"
              >
                Learn More →
              </button>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.3 }}
              className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-8 border border-pink-500/20 hover:border-pink-500/40 transition-all"
            >
              <BarChart3 className="h-12 w-12 text-pink-400 mb-6" />
              <h3 className="text-2xl font-bold mb-4">Progress Tracking</h3>
              <p className="text-gray-300 mb-6">
                Track your learning journey with detailed analytics, 
                daily streaks, and personalized course recommendations.
              </p>
              <button
                onClick={() => scrollToSection('dashboard-section')}
                className="text-pink-400 hover:text-pink-300 font-semibold"
              >
                Learn More →
              </button>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Quiz Section */}
      <section id="quiz-section" className="py-20 px-6 bg-slate-800/30 relative">
        <div className="container mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
            >
              <div className="flex items-center space-x-3 mb-6">
                <Target className="h-8 w-8 text-purple-400" />
                <h3 className="text-3xl font-bold">Intelligent Quizzing</h3>
              </div>
              <p className="text-gray-300 mb-6 text-lg">
                Our AI creates personalized quizzes based on your profession and chosen topics. 
                Progress through questions one at a time with instant feedback and recommendations.
              </p>
              <ul className="space-y-3 mb-8">
                <li className="flex items-center space-x-3">
                  <Zap className="h-5 w-5 text-purple-400" />
                  <span>Profession-based question generation</span>
                </li>
                <li className="flex items-center space-x-3">
                  <Zap className="h-5 w-5 text-purple-400" />
                  <span>Difficulty adaptation</span>
                </li>
                <li className="flex items-center space-x-3">
                  <Zap className="h-5 w-5 text-purple-400" />
                  <span>Instant scoring and feedback</span>
                </li>
              </ul>
              <Link
                to="/quiz"
                className="inline-flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 rounded-lg font-semibold transform hover:scale-105 transition-all"
              >
                <Brain className="h-5 w-5" />
                <span>Start Quiz</span>
              </Link>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="bg-gradient-to-br from-purple-900/50 to-pink-900/50 p-8 rounded-xl border border-purple-500/20"
            >
              <div className="space-y-4">
                <div className="bg-slate-800/80 p-4 rounded-lg border-l-4 border-purple-400">
                  <h4 className="font-semibold text-purple-400 mb-2">Question 1 of 5</h4>
                  <p className="text-gray-300">What is the primary benefit of using React hooks?</p>
                </div>
                <div className="space-y-2">
                  <div className="bg-slate-700/50 p-3 rounded-lg hover:bg-slate-600/50 cursor-pointer transition-colors">
                    A) Better performance
                  </div>
                  <div className="bg-slate-700/50 p-3 rounded-lg hover:bg-slate-600/50 cursor-pointer transition-colors">
                    B) Simpler component logic
                  </div>
                  <div className="bg-slate-700/50 p-3 rounded-lg hover:bg-slate-600/50 cursor-pointer transition-colors">
                    C) Enhanced styling capabilities
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Chat Section */}
      <section id="chat-section" className="py-20 px-6 relative">
        <div className="container mx-auto">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="order-2 md:order-1 bg-gradient-to-br from-slate-900 to-purple-900 p-8 rounded-xl border border-cyan-500/20 relative overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-400/10 to-blue-400/10" />
              <div className="relative space-y-4">
                <div className="flex justify-end">
                  <div className="bg-cyan-600 text-white p-3 rounded-l-lg rounded-tr-lg max-w-xs">
                    How do I implement authentication in React?
                  </div>
                </div>
                <div className="flex justify-start">
                  <div className="bg-slate-700 text-white p-3 rounded-r-lg rounded-tl-lg max-w-xs">
                    <div className="flex items-center space-x-2 mb-2">
                      <Sparkles className="h-4 w-4 text-cyan-400" />
                      <span className="text-cyan-400 font-semibold">Cosmic AI</span>
                    </div>
                    Great question! For React authentication, I recommend using Firebase Auth or Auth0...
                  </div>
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="order-1 md:order-2"
            >
              <div className="flex items-center space-x-3 mb-6">
                <MessageCircle className="h-8 w-8 text-cyan-400" />
                <h3 className="text-3xl font-bold">Cosmic AI Assistant</h3>
              </div>
              <p className="text-gray-300 mb-6 text-lg">
                Get instant help from our AI assistant in a beautiful cosmic chat environment. 
                Ask questions, get explanations, and receive personalized learning guidance.
              </p>
              <ul className="space-y-3 mb-8">
                <li className="flex items-center space-x-3">
                  <Sparkles className="h-5 w-5 text-cyan-400" />
                  <span>24/7 AI assistance</span>
                </li>
                <li className="flex items-center space-x-3">
                  <Sparkles className="h-5 w-5 text-cyan-400" />
                  <span>Conversation history</span>
                </li>
                <li className="flex items-center space-x-3">
                  <Sparkles className="h-5 w-5 text-cyan-400" />
                  <span>Cosmic-themed interface</span>
                </li>
              </ul>
              <Link
                to="/chatbot"
                className="inline-flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 rounded-lg font-semibold transform hover:scale-105 transition-all"
              >
                <MessageCircle className="h-5 w-5" />
                <span>Start Chatting</span>
              </Link>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Dashboard Section */}
      <section id="dashboard-section" className="py-20 px-6 bg-slate-800/30 relative">
        <div className="container mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-16"
          >
            <h3 className="text-3xl font-bold mb-6 bg-gradient-to-r from-pink-400 to-purple-400 bg-clip-text text-transparent">
              Track Your Cosmic Journey
            </h3>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Monitor your progress, maintain learning streaks, and discover personalized course recommendations
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="bg-gradient-to-br from-slate-800/80 to-purple-800/80 p-8 rounded-xl border border-pink-500/20 max-w-2xl mx-auto"
          >
            <div className="grid grid-cols-3 gap-6 text-center">
              <div>
                <div className="text-3xl font-bold text-pink-400">127</div>
                <div className="text-gray-400">Quiz Score Avg</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-purple-400">15</div>
                <div className="text-gray-400">Day Streak</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-cyan-400">8</div>
                <div className="text-gray-400">Courses Found</div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Home;