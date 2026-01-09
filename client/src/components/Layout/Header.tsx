import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../../context/AuthContext';
import { Sparkles, User, LogOut } from 'lucide-react';

const Header: React.FC = () => {
  const { user, logout } = useAuth();
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gradient-to-r from-slate-900 via-purple-900 to-slate-900 backdrop-blur-sm border-b border-purple-500/20 sticky top-0 z-50"
    >
      <nav className="container mx-auto px-6 py-4 flex items-center justify-between">
        <Link to="/" className="flex items-center space-x-2">
          <Sparkles className="h-8 w-8 text-purple-400" />
          <span className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            CosmicLearn
          </span>
        </Link>

        <div className="hidden md:flex items-center space-x-6">
          <Link
            to="/"
            className={`hover:text-purple-400 transition-colors ${
              isActive('/') ? 'text-purple-400' : 'text-white'
            }`}
          >
            Home
          </Link>
          <Link
            to="/quiz"
            className={`hover:text-purple-400 transition-colors ${
              isActive('/quiz') ? 'text-purple-400' : 'text-white'
            }`}
          >
            Quiz
          </Link>
          <Link
            to="/chatbot"
            className={`hover:text-purple-400 transition-colors ${
              isActive('/chatbot') ? 'text-purple-400' : 'text-white'
            }`}
          >
            AI Chat
          </Link>
          {user && (
            <Link
              to="/dashboard"
              className={`hover:text-purple-400 transition-colors ${
                isActive('/dashboard') ? 'text-purple-400' : 'text-white'
              }`}
            >
              Dashboard
            </Link>
          )}
        </div>

        {/* <div className="flex items-center space-x-4">
          {user ? (
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2 text-white">
                <User className="h-5 w-5" />
                <span className="hidden md:inline">{user.email}</span>
              </div>
              <button
                onClick={logout}
                className="flex items-center space-x-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
              >
                <LogOut className="h-4 w-4" />
                <span className="hidden md:inline">Logout</span>
              </button>
            </div>
          ) : (
            <Link
              to="/auth"
              className="px-6 py-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg transition-all transform hover:scale-105"
            >
              Sign In
            </Link>
          )}
        </div> */}
      </nav>
    </motion.header>
  );
};

export default Header;