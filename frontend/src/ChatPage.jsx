import { useEffect, useRef, useState } from "react";

const API_URL = "http://127.0.0.1:8000/query";

function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState("semantic");
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
        body: JSON.stringify({ query, top_k: 5, mode }),
      });
      if (!res.ok) throw new Error(`Request failed (${res.status})`);
      const data = await res.json();

      if (mode === "keyword") {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", mode: "keyword", results: data.sources },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", text: data.answer, sources: data.sources },
        ]);
      }
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
      <div className="chat-toolbar">
        <div className="mode-toggle" role="tablist" aria-label="Mode de recherche">
          <button
            type="button"
            role="tab"
            aria-selected={mode === "semantic"}
            className={`mode-button ${mode === "semantic" ? "active" : ""}`}
            onClick={() => setMode("semantic")}
          >
            Analyse IA
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "keyword"}
            className={`mode-button ${mode === "keyword" ? "active" : ""}`}
            onClick={() => setMode("keyword")}
          >
            Mot-clé
          </button>
        </div>

        {messages.length > 0 && (
          <button type="button" className="reset-button" onClick={resetConversation}>
            Nouvelle conversation
          </button>
        )}
      </div>

      <div className="chat-window">
        {messages.length === 0 && !loading && (
          <div className="empty-state">
            {mode === "semantic" ? (
              <>
                Pose une question sur le Plan Comptable Général.
                <br />
                Exemple : « Qu'est-ce qu'une provision ? »
              </>
            ) : (
              <>
                Cherche un mot ou une expression dans le PCG.
                <br />
                Exemple : « provision pour risques »
              </>
            )}
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            {m.mode === "keyword" ? (
              <div className="bubble keyword-results">
                {m.results.length === 0 ? (
                  <p>Aucun résultat pour cette recherche.</p>
                ) : (
                  m.results.map((r) => (
                    <div className="keyword-result" key={`${r.article_number}-${r.rank_lexical}`}>
                      <div className="keyword-result-header">
                        <span className="source-chip">Art. {r.article_number}</span>
                        {r.article_title && <span className="keyword-result-title">{r.article_title}</span>}
                      </div>
                      <p>{r.text}</p>
                    </div>
                  ))
                )}
              </div>
            ) : (
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
            )}
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
          placeholder={mode === "semantic" ? "Pose une question sur le PCG..." : "Cherche un mot-clé dans le PCG..."}
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
