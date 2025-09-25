import React from 'react';
import { Sparkles } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-gradient-to-r from-slate-900 via-purple-900 to-slate-900 border-t border-purple-500/20 py-8">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row items-center justify-between">
          <div className="flex items-center space-x-2 mb-4 md:mb-0">
            <Sparkles className="h-6 w-6 text-purple-400" />
            <span className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              CosmicLearn
            </span>
          </div>
          <div className="text-gray-400 text-sm">
            © 2025 CosmicLearn. Powered by AI and cosmic energy.
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;