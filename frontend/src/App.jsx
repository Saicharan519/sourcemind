import { Routes, Route, Link, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import Dashboard from "./pages/Dashboard.jsx";
import ChatPage from "./pages/ChatPage.jsx";
import GlobalChatPage from "./pages/GlobalChatPage.jsx";
import SourceDetailPage from "./pages/SourceDetailPage.jsx";

function NavLink({ to, children }) {
  const { pathname } = useLocation();
  const active =
    to === "/" ? pathname === "/" : pathname.startsWith(to);
  return (
    <Link to={to} className="relative px-1 py-1 text-sm text-ink-soft transition-colors hover:text-ink">
      <span className={active ? "text-ink" : ""}>{children}</span>
      {active && (
        <motion.span
          layoutId="nav-underline"
          className="absolute -bottom-0.5 left-0 right-0 h-[2px] rounded-full bg-emerald"
          transition={{ type: "spring", stiffness: 400, damping: 32 }}
        />
      )}
    </Link>
  );
}

export default function App() {
  const location = useLocation();
  return (
    <div className="min-h-screen">
      <div className="atmosphere" />

      <header className="sticky top-0 z-30 border-b border-line/70 bg-paper/70 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <Link to="/" className="group flex items-center gap-2.5">
            <motion.span
              initial={{ rotate: -8, scale: 0.9 }}
              animate={{ rotate: 0, scale: 1 }}
              transition={{ type: "spring", stiffness: 260, damping: 18 }}
              className="grid h-9 w-9 place-items-center rounded-xl bg-ink text-paper shadow-card"
            >
              <span className="font-display text-lg leading-none">S</span>
            </motion.span>
            <span className="font-display text-xl font-semibold tracking-tight text-ink sm:text-2xl">
              Source<span className="italic text-emerald">Mind</span>
            </span>
          </Link>
          <nav className="flex items-center gap-4 sm:gap-6">
            <NavLink to="/">Library</NavLink>
            <NavLink to="/global">
              <span className="whitespace-nowrap">Global chat</span>
            </NavLink>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-10">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
          >
            <Routes location={location}>
              <Route path="/" element={<Dashboard />} />
              <Route path="/chat/:sourceId" element={<ChatPage />} />
              <Route path="/source/:sourceId" element={<SourceDetailPage />} />
              <Route path="/global" element={<GlobalChatPage />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
