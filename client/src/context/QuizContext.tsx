import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";

interface Question {
  id: string;
  question: string;
  options: string[];
  // backend uses `correctAnswer` (either an index or the correct option string)
  correctAnswer: number | string;
}

interface QuizState {
  profession: string;
  topic: string;
  difficulty: string;
  questions: Question[];
  currentQuestionIndex: number;
  answers: Record<string, string>;
  score: number;
  isCompleted: boolean;
}

interface QuizContextType {
  quizState: QuizState;
  setQuizConfig: (
    profession: string,
    topic: string,
    difficulty: string
  ) => void;
  setQuestions: (questions: Question[]) => void;
  answerQuestion: (questionId: string, answer: string) => void;
  nextQuestion: () => void;
  previousQuestion: () => void;
  resetQuiz: () => void;
  calculateScore: () => number;
  completeQuiz: () => void;
}

const initialState: QuizState = {
  profession: "",
  topic: "",
  difficulty: "medium",
  questions: [],
  currentQuestionIndex: 0,
  answers: {},
  score: 0,
  isCompleted: false,
};

const QuizContext = createContext<QuizContextType | undefined>(undefined);

export const useQuiz = () => {
  const context = useContext(QuizContext);
  if (context === undefined) {
    throw new Error("useQuiz must be used within a QuizProvider");
  }
  return context;
};

interface QuizProviderProps {
  children: ReactNode;
}

export const QuizProvider: React.FC<QuizProviderProps> = ({ children }) => {
  const [quizState, setQuizState] = useState<QuizState>(initialState);

  // Save quiz state to localStorage on changes
  useEffect(() => {
    if (quizState.questions.length > 0) {
      localStorage.setItem("currentQuiz", JSON.stringify(quizState));
    }
  }, [quizState]);

  // Load quiz state from localStorage on mount
  useEffect(() => {
    // Clear any in-progress quiz on startup so a restarted frontend starts fresh.
    // This prevents resuming a partially completed quiz after restarting the dev server.
    try {
      localStorage.removeItem("currentQuiz");
    } catch (err) {
      console.warn("Could not clear saved quiz on startup:", err);
    }
  }, []);

  const setQuizConfig = (
    profession: string,
    topic: string,
    difficulty: string
  ) => {
    setQuizState((prev) => ({
      ...prev,
      profession,
      topic,
      difficulty,
      currentQuestionIndex: 0,
      answers: {},
      score: 0,
      isCompleted: false,
    }));
  };

  const setQuestions = (questions: Question[]) => {
    setQuizState((prev) => ({
      ...prev,
      questions,
    }));
  };

  const answerQuestion = (questionId: string, answer: string) => {
    setQuizState((prev) => ({
      ...prev,
      answers: {
        ...prev.answers,
        [questionId]: answer,
      },
    }));
  };

  const nextQuestion = () => {
    setQuizState((prev) => ({
      ...prev,
      currentQuestionIndex: Math.min(
        prev.currentQuestionIndex + 1,
        prev.questions.length - 1
      ),
    }));
  };

  const previousQuestion = () => {
    setQuizState((prev) => ({
      ...prev,
      currentQuestionIndex: Math.max(prev.currentQuestionIndex - 1, 0),
    }));
  };

  const calculateScore = () => {
    let correctAnswers = 0;
    quizState.questions.forEach((question) => {
      const userAnswer = quizState.answers[question.id];
      if (userAnswer == null) return;

      // If backend stored correctAnswer as an index, resolve to the option string
      let correctOption: string | undefined;
      if (typeof question.correctAnswer === "number") {
        correctOption = question.options[question.correctAnswer];
      } else if (typeof question.correctAnswer === "string") {
        correctOption = question.correctAnswer;
      }

      if (correctOption && userAnswer === correctOption) {
        correctAnswers++;
      }
    });

    const total = quizState.questions.length || 1;
    const score = Math.round((correctAnswers / total) * 100);
    return score;
  };

  const completeQuiz = () => {
    const score = calculateScore();
    setQuizState((prev) => ({
      ...prev,
      score,
      isCompleted: true,
    }));
  };

  const resetQuiz = () => {
    setQuizState(initialState);
    localStorage.removeItem("currentQuiz");
  };

  const value = {
    quizState,
    setQuizConfig,
    setQuestions,
    answerQuestion,
    nextQuestion,
    previousQuestion,
    resetQuiz,
    calculateScore,
    completeQuiz,
  };

  return <QuizContext.Provider value={value}>{children}</QuizContext.Provider>;
};
