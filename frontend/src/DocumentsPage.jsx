import { useEffect, useRef, useState } from "react";

const DOCUMENTS_URL = "http://127.0.0.1:8000/documents";

function groupByArticle(chunks) {
  const groups = [];
  const byKey = new Map();
  for (const chunk of chunks) {
    const key = `${chunk.article_number}::${chunk.article_title}`;
    let group = byKey.get(key);
    if (!group) {
      group = {
        article_number: chunk.article_number,
        article_title: chunk.article_title,
        subChunks: [],
      };
      byKey.set(key, group);
      groups.push(group);
    }
    group.subChunks.push(chunk);
  }
  return groups;
}

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} Ko`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} Mo`;
}

const STATUS_LABELS = {
  pending: "En attente",
  processing: "Traitement...",
  done: "Pret",
  failed: "Echec",
};

function DocumentsPage() {
  const [documents, setDocuments] = useState([]);
  const [pendingFile, setPendingFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedId, setSelectedId] = useState(null);
  const [chunks, setChunks] = useState(null);
  const [chunksLoading, setChunksLoading] = useState(false);
  const [expanded, setExpanded] = useState(() => new Set());
  const fileInputRef = useRef(null);

  async function fetchDocuments() {
    const res = await fetch(DOCUMENTS_URL);
    if (!res.ok) throw new Error(`Request failed (${res.status})`);
    setDocuments(await res.json());
  }

  useEffect(() => {
    fetchDocuments().catch(() => {});
  }, []);

  useEffect(() => {
    const hasPending = documents.some(
      (d) => d.status === "pending" || d.status === "processing"
    );
    if (!hasPending) return;

    const interval = setInterval(() => {
      fetchDocuments().catch(() => {});
    }, 4000);
    return () => clearInterval(interval);
  }, [documents]);

  function pickFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Seuls les fichiers .pdf sont acceptes.");
      return;
    }
    setUploadError(null);
    setPendingFile(file);
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    pickFile(e.dataTransfer.files?.[0]);
  }

  async function handleUpload(e) {
    e.preventDefault();
    if (!pendingFile) return;

    setUploading(true);
    setUploadError(null);
    try {
      const formData = new FormData();
      formData.append("file", pendingFile);
      const res = await fetch(DOCUMENTS_URL, { method: "POST", body: formData });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      setPendingFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      await fetchDocuments();
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function deleteDocument(doc, e) {
    e.stopPropagation();
    if (!window.confirm(`Supprimer "${doc.filename}" et tous ses chunks ?`)) return;

    try {
      const res = await fetch(`${DOCUMENTS_URL}/${doc.id}`, { method: "DELETE" });
      if (!res.ok && res.status !== 204) throw new Error(`Request failed (${res.status})`);
      if (selectedId === doc.id) {
        setSelectedId(null);
        setChunks(null);
      }
      await fetchDocuments();
    } catch (err) {
      setUploadError(err.message);
    }
  }

  async function selectDocument(doc) {
    if (doc.status !== "done") return;
    setSelectedId(doc.id);
    setChunks(null);
    setChunksLoading(true);
    setExpanded(new Set());
    try {
      const res = await fetch(`${DOCUMENTS_URL}/${doc.id}/chunks`);
      if (!res.ok) throw new Error(`Request failed (${res.status})`);
      setChunks(await res.json());
    } catch {
      setChunks([]);
    } finally {
      setChunksLoading(false);
    }
  }

  function toggleArticle(key) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  const groups = chunks ? groupByArticle(chunks) : [];

  return (
    <div className="documents-app">
      <form className="upload-form" onSubmit={handleUpload}>
        <label
          className={`dropzone ${isDragging ? "dragging" : ""} ${pendingFile ? "has-file" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept=".pdf"
            ref={fileInputRef}
            disabled={uploading}
            onChange={(e) => pickFile(e.target.files?.[0])}
          />
          {pendingFile ? (
            <div className="dropzone-file">
              <span className="dropzone-filename">{pendingFile.name}</span>
              <span className="dropzone-filesize">{formatSize(pendingFile.size)}</span>
            </div>
          ) : (
            <div className="dropzone-hint">
              <span className="dropzone-title">Glisse un PDF ici ou clique pour choisir</span>
              <span className="dropzone-subtitle">Fichier .pdf uniquement</span>
            </div>
          )}
        </label>
        <button type="submit" disabled={uploading || !pendingFile}>
          {uploading ? "Envoi en cours..." : "Ajouter le document"}
        </button>
      </form>
      {uploadError && <div className="upload-error">{uploadError}</div>}

      <div className="documents-layout">
        <div className="documents-list">
          {documents.length === 0 && (
            <div className="empty-state">
              Aucun document pour le moment.
              <br />
              Ajoute un PDF ci-dessus pour commencer.
            </div>
          )}
          {documents.map((doc) => (
            <div key={doc.id} className={`document-row ${doc.id === selectedId ? "selected" : ""}`}>
              <button
                className="document-row-main"
                onClick={() => selectDocument(doc)}
                disabled={doc.status !== "done"}
                title={doc.status !== "done" ? STATUS_LABELS[doc.status] : "Voir les chunks"}
              >
                <span className="document-filename">{doc.filename}</span>
                <span className={`status-badge status-${doc.status}`}>
                  {(doc.status === "pending" || doc.status === "processing") && (
                    <span className="status-spinner" />
                  )}
                  {STATUS_LABELS[doc.status] ?? doc.status}
                </span>
              </button>
              <button
                className="document-delete"
                onClick={(e) => deleteDocument(doc, e)}
                title="Supprimer ce document"
                aria-label="Supprimer ce document"
              >
                &times;
              </button>
            </div>
          ))}
        </div>

        <div className="document-chunks">
          {!selectedId && (
            <div className="empty-state">
              Selectionne un document a gauche pour parcourir ses articles et chunks.
            </div>
          )}
          {selectedId && chunksLoading && (
            <div className="empty-state">
              <span className="status-spinner large" />
              Chargement des chunks...
            </div>
          )}
          {selectedId && !chunksLoading && groups.length === 0 && (
            <div className="empty-state">Aucun chunk trouve pour ce document.</div>
          )}
          {groups.map((group) => {
            const key = `${group.article_number}::${group.article_title}`;
            const isOpen = expanded.has(key);
            return (
              <div className="article-group" key={key}>
                <button className="article-header" onClick={() => toggleArticle(key)}>
                  <span>
                    Art. {group.article_number} -- {group.article_title}
                  </span>
                  <span className="article-count">{group.subChunks.length}</span>
                </button>
                {isOpen && (
                  <div className="article-body">
                    {group.subChunks.map((c, i) => (
                      <div className="sub-chunk" key={i}>
                        {c.sub_chunk !== null && (
                          <div className="sub-chunk-label">
                            {c.sub_chunk + 1}/{c.sub_chunk_total}
                          </div>
                        )}
                        <p>{c.text_content}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default DocumentsPage;
