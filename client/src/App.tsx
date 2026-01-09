import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

import { AuthProvider } from "./context/AuthContext";
import { QuizProvider } from "./context/QuizContext";
import Header from "./components/Layout/Header";
import Footer from "./components/Layout/Footer";
import Home from "./pages/Home";
import Quiz from "./pages/Quiz";
import QuizResult from "./pages/QuizResult";
import Chatbot from "./pages/Chatbot";
// import Auth from './pages/Auth'; // COMMENTED OUT: Sign-in functionality disabled
// import Dashboard from './pages/Dashboard'; // COMMENTED OUT: Dashboard requires auth

// COMMENTED OUT: ProtectedRoute component - Auth functionality disabled
/*
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white flex items-center justify-center">
        <LoadingSpinner />
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/auth" />;
  }

  return <>{children}</>;
};
*/

const AppRoutes: React.FC = () => {
  return (
    <div className="flex flex-col min-h-screen">
      <Header />
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/quiz" element={<Quiz />} />
          <Route path="/quiz/result" element={<QuizResult />} />
          <Route path="/chatbot" element={<Chatbot />} />
          {/* COMMENTED OUT: Auth route disabled */}
          {/* <Route path="/auth" element={<Auth />} /> */}
          {/* COMMENTED OUT: Dashboard route disabled */}
          {/* <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            }
          /> */}
        </Routes>
      </main>
      <Footer />

      <ToastContainer
        position="top-right"
        autoClose={3000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="dark"
        toastStyle={{
          backgroundColor: "#1e293b",
          color: "#f8fafc",
          border: "1px solid #7c3aed",
        }}
      />
    </div>
  );
};

const App: React.FC = () => {
  return (
    <AuthProvider>
      <QuizProvider>
        <Router>
          <AppRoutes />
        </Router>
      </QuizProvider>
    </AuthProvider>
  );
};

export default App;
