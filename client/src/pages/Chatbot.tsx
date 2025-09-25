import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Sparkles, User, Bot } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { apiHelpers } from '../config/api';
import { toast } from 'react-toastify';
import StarBackground from '../components/UI/StarBackground';

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
}

const Chatbot: React.FC = () => {
  const { user } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [conversationId] = useState(() => {
    const existing = localStorage.getItem('conversationId');
    if (existing) return existing;
    const newId = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem('conversationId', newId);
    return newId;
  });
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Load chat history from localStorage
    const savedMessages = localStorage.getItem(`chat_${conversationId}`);
    if (savedMessages) {
      try {
        const parsed = JSON.parse(savedMessages);
        setMessages(parsed.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        })));
      } catch (error) {
        console.error('Failed to load chat history:', error);
      }
    } else {
      // Add welcome message
      setMessages([{
        id: 'welcome',
        content: "🌟 Welcome to the cosmic realm of learning! I'm your AI assistant, ready to help you explore the universe of knowledge. Ask me anything about programming, science, or any topic you're curious about!",
        isUser: false,
        timestamp: new Date()
      }]);
    }
  }, [conversationId]);

  useEffect(() => {
    // Save messages to localStorage whenever they change
    if (messages.length > 0) {
      localStorage.setItem(`chat_${conversationId}`, JSON.stringify(messages));
    }
  }, [messages, conversationId]);

  useEffect(() => {
    // Auto-scroll to bottom
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage: Message = {
      id: `msg_${Date.now()}_user`,
      content: inputMessage.trim(),
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsTyping(true);

    try {
      const response = await apiHelpers.sendChatMessage(
        userMessage.content,
        conversationId,
        user?.uid || null
      );

      const aiMessage: Message = {
        id: `msg_${Date.now()}_ai`,
        content: response.response || 'I apologize, but I encountered an issue. Please try again.',
        isUser: false,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Failed to send message:', error);
      toast.error('Failed to send message. Please try again.');
      
      const errorMessage: Message = {
        id: `msg_${Date.now()}_error`,
        content: 'I apologize, but I\'m having trouble connecting to the cosmic network. Please check your connection and try again.',
        isUser: false,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white relative overflow-hidden">
      <StarBackground />
      
      <div className="container mx-auto px-4 py-6 h-screen flex flex-col">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-center space-x-3 mb-6"
        >
          <div className="relative">
            <Bot className="h-8 w-8 text-cyan-400" />
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="absolute -inset-1 bg-cyan-400/20 rounded-full"
            />
          </div>
          <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
            Cosmic AI Assistant
          </h1>
        </motion.div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto mb-4 bg-slate-800/30 backdrop-blur-sm rounded-xl border border-purple-500/20 p-4">
          <AnimatePresence>
            {messages.map((message, index) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                className={`flex items-start space-x-3 mb-6 ${
                  message.isUser ? 'flex-row-reverse space-x-reverse' : ''
                }`}
              >
                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                  message.isUser 
                    ? 'bg-gradient-to-r from-purple-600 to-pink-600' 
                    : 'bg-gradient-to-r from-cyan-600 to-blue-600'
                }`}>
                  {message.isUser ? (
                    <User className="h-4 w-4" />
                  ) : (
                    <Sparkles className="h-4 w-4" />
                  )}
                </div>
                
                <div className={`flex-1 max-w-xs sm:max-w-md lg:max-w-lg xl:max-w-xl ${
                  message.isUser ? 'text-right' : 'text-left'
                }`}>
                  <motion.div
                    initial={{ scale: 0.9 }}
                    animate={{ scale: 1 }}
                    className={`inline-block p-4 rounded-lg ${
                      message.isUser
                        ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white'
                        : 'bg-slate-700/80 text-gray-100 border border-cyan-500/20'
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  </motion.div>
                  <p className={`text-xs text-gray-400 mt-1 ${
                    message.isUser ? 'text-right' : 'text-left'
                  }`}>
                    {formatTime(message.timestamp)}
                  </p>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Typing Indicator */}
          {isTyping && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-start space-x-3 mb-6"
            >
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-r from-cyan-600 to-blue-600 flex items-center justify-center">
                <Sparkles className="h-4 w-4" />
              </div>
              <div className="bg-slate-700/80 p-4 rounded-lg border border-cyan-500/20">
                <div className="flex items-center space-x-1">
                  <div className="flex space-x-1">
                    {[0, 1, 2].map((i) => (
                      <motion.div
                        key={i}
                        className="w-2 h-2 bg-cyan-400 rounded-full"
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{
                          duration: 0.8,
                          repeat: Infinity,
                          delay: i * 0.2,
                        }}
                      />
                    ))}
                  </div>
                  <span className="text-gray-400 text-sm ml-2">AI is thinking...</span>
                </div>
              </div>
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-slate-800/50 backdrop-blur-sm rounded-xl border border-purple-500/20 p-4"
        >
          <div className="flex items-center space-x-3">
            <input
              ref={inputRef}
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me anything about the cosmos of knowledge..."
              className="flex-1 bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-cyan-400 transition-colors"
              disabled={isTyping}
            />
            <motion.button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || isTyping}
              className="px-4 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Send className="h-5 w-5" />
            </motion.button>
          </div>
          
          {!user && (
            <p className="text-gray-400 text-sm mt-2">
              💡 Tip: Sign in to save your conversation history across devices!
            </p>
          )}
        </motion.div>
      </div>
    </div>
  );
};

export default Chatbot;