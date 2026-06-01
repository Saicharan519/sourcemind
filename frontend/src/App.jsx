import { Routes, Route, Link } from "react-router-dom";
import Dashboard from "./pages/Dashboard.jsx";
import ChatPage from "./pages/ChatPage.jsx";
import GlobalChatPage from "./pages/GlobalChatPage.jsx";
import SourceDetailPage from "./pages/SourceDetailPage.jsx";

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-border bg-surface px-6 py-4 flex items-center justify-between">
        <Link to="/" className="text-xl font-bold tracking-tight">
          🧠 <span className="text-accent-glow">Source</span>Mind
        </Link>
        <nav className="flex gap-4 text-sm text-muted">
          <Link to="/" className="hover:text-white">
            Sources
          </Link>
          <Link to="/global" className="hover:text-white">
            Global chat
          </Link>
        </nav>
      </header>
      <main className="p-6">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/chat/:sourceId" element={<ChatPage />} />
          <Route path="/source/:sourceId" element={<SourceDetailPage />} />
          <Route path="/global" element={<GlobalChatPage />} />
        </Routes>
      </main>
    </div>
  );
}
