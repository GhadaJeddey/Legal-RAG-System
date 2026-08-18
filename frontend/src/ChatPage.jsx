import { useEffect, useRef, useState } from "react";

const API_URL = "http://127.0.0.1:8000/query";

function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  function resetConversation() {
    setMessages([]);
    setInput("");
    inputRef.current?.focus();
  }

  async function sendQuery(e) {
    e.preventDefault();
    const query = input.trim();
    if (!query || loading) return;

    setMessages((prev) => [...prev, { role: "user", text: query }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
      });
      if (!res.ok) throw new Error(`Request failed (${res.status})`);
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.answer, sources: data.sources },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "error", text: `Erreur: ${err.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-app">
      {messages.length > 0 && (
        <div className="chat-toolbar">
          <button type="button" className="reset-button" onClick={resetConversation}>
            Nouvelle conversation
          </button>
        </div>
      )}

      <div className="chat-window">
        {messages.length === 0 && !loading && (
          <div className="empty-state">
            Pose une question sur le Plan Comptable Général.
            <br />
            Exemple : « Qu'est-ce qu'une provision ? »
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            <div className="bubble">
              <p>{m.text}</p>
              {m.sources && m.sources.length > 0 && (
                <div className="sources">
                  {m.sources.map((s) => (
                    <span
                      className="source-chip"
                      key={s.article_number}
                      title={`${s.article_title ?? ""} -- pertinence ${Math.round(s.similarity * 100)}%`}
                    >
                      Art. {s.article_number}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="message assistant">
            <div className="bubble typing">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <form className="chat-input" onSubmit={sendQuery}>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Pose une question sur le PCG..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Envoyer
        </button>
      </form>
    </div>
  );
}

export default ChatPage;
