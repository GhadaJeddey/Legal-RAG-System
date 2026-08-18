import { useState } from "react";
import "./App.css";
import ChatPage from "./ChatPage";
import DocumentsPage from "./DocumentsPage";

function App() {
  const [tab, setTab] = useState("chat");

  return (
    <div className={`app-shell ${tab === "documents" ? "wide" : ""}`}>
      <header className="chat-header">
        <h1>PCG RAG Assistant</h1>
        <p className="subtitle">Questions-réponses sur le Plan Comptable Général</p>
      </header>

      <div className="tab-bar" role="tablist">
        <button
          role="tab"
          aria-selected={tab === "chat"}
          className={`tab-button ${tab === "chat" ? "active" : ""}`}
          onClick={() => setTab("chat")}
        >
          Chat
        </button>
        <button
          role="tab"
          aria-selected={tab === "documents"}
          className={`tab-button ${tab === "documents" ? "active" : ""}`}
          onClick={() => setTab("documents")}
        >
          Documents
        </button>
      </div>

      {tab === "chat" ? <ChatPage /> : <DocumentsPage />}
    </div>
  );
}

export default App;
