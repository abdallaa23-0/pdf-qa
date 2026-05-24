import { useState, useRef, useEffect } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── Minimal design system ─────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0d0d0d;
    --surface: #151515;
    --border: #2a2a2a;
    --accent: #e8ff47;
    --accent-dim: #b8cc2f;
    --text: #e8e8e8;
    --muted: #666;
    --danger: #ff4747;
    --radius: 6px;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
  }

  body { background: var(--bg); color: var(--text); font-family: var(--sans); min-height: 100vh; }

  .app { display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }

  /* ── Sidebar ── */
  .sidebar {
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
    padding: 24px 16px;
    gap: 24px;
  }
  .logo { font-family: var(--mono); font-size: 13px; color: var(--accent); letter-spacing: 0.08em; }
  .logo span { color: var(--muted); }

  .upload-zone {
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    padding: 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
    position: relative;
  }
  .upload-zone:hover, .upload-zone.drag { border-color: var(--accent); background: rgba(232,255,71,0.03); }
  .upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; }
  .upload-zone p { font-size: 12px; color: var(--muted); line-height: 1.6; }
  .upload-zone strong { color: var(--text); font-size: 13px; display: block; margin-bottom: 4px; }

  .doc-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
  .doc-list-label { font-family: var(--mono); font-size: 10px; color: var(--muted); letter-spacing: 0.12em; margin-bottom: 4px; }

  .doc-item {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 10px;
    border-radius: var(--radius);
    border: 1px solid transparent;
    cursor: pointer;
    transition: all 0.1s;
    font-size: 12px;
  }
  .doc-item:hover { background: rgba(255,255,255,0.04); border-color: var(--border); }
  .doc-item.active { background: rgba(232,255,71,0.07); border-color: var(--accent); color: var(--accent); }
  .doc-item .doc-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .doc-item .del { color: var(--muted); font-size: 14px; line-height: 1; padding: 2px; }
  .doc-item .del:hover { color: var(--danger); }

  /* ── Main ── */
  .main { display: flex; flex-direction: column; height: 100vh; }

  .topbar {
    padding: 16px 28px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
  }
  .topbar-title { font-family: var(--mono); font-size: 12px; color: var(--muted); }
  .topbar-doc { font-size: 13px; color: var(--text); }

  .chat { flex: 1; overflow-y: auto; padding: 28px; display: flex; flex-direction: column; gap: 20px; }

  .empty-state {
    flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 12px; color: var(--muted); font-size: 13px; text-align: center;
  }
  .empty-state .icon { font-size: 40px; opacity: 0.3; }
  .empty-state p { max-width: 260px; line-height: 1.6; }

  .msg { display: flex; flex-direction: column; gap: 6px; max-width: 720px; }
  .msg.user { align-self: flex-end; align-items: flex-end; }
  .msg.assistant { align-self: flex-start; }

  .bubble {
    padding: 12px 16px;
    border-radius: var(--radius);
    font-size: 14px;
    line-height: 1.65;
  }
  .msg.user .bubble { background: var(--accent); color: #111; font-weight: 500; }
  .msg.assistant .bubble { background: var(--surface); border: 1px solid var(--border); }

  .sources { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
  .source-chip {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 8px;
    cursor: pointer;
    transition: all 0.1s;
  }
  .source-chip:hover { color: var(--accent); border-color: var(--accent); }

  .source-tooltip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px;
    font-size: 12px;
    color: var(--muted);
    line-height: 1.6;
    max-width: 480px;
    margin-top: 4px;
  }

  .thinking { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 13px; padding: 12px 16px; }
  .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); animation: pulse 1.2s infinite; }
  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes pulse { 0%,80%,100% { opacity: 0.2; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1); } }

  /* ── Input bar ── */
  .inputbar {
    padding: 16px 28px 24px;
    border-top: 1px solid var(--border);
    display: flex; gap: 10px; align-items: flex-end;
  }
  .inputbar textarea {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    padding: 12px 14px;
    resize: none;
    outline: none;
    max-height: 160px;
    transition: border-color 0.15s;
    line-height: 1.5;
  }
  .inputbar textarea:focus { border-color: var(--accent); }
  .inputbar textarea::placeholder { color: var(--muted); }

  .send-btn {
    background: var(--accent);
    color: #111;
    border: none;
    border-radius: var(--radius);
    padding: 12px 20px;
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.1s;
    height: 44px;
  }
  .send-btn:hover { background: var(--accent-dim); }
  .send-btn:disabled { opacity: 0.35; cursor: not-allowed; }

  .toast {
    position: fixed; bottom: 24px; right: 24px;
    background: var(--danger); color: #fff;
    padding: 10px 16px; border-radius: var(--radius);
    font-size: 13px; z-index: 999;
    animation: slideIn 0.2s ease;
  }
  @keyframes slideIn { from { transform: translateY(8px); opacity:0; } to { transform: none; opacity:1; } }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
`;

// ── Components ────────────────────────────────────────────────────────────────

function SourceChip({ source }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button className="source-chip" onClick={() => setOpen(o => !o)}>
        📄 {source.filename} · {Math.round(source.score * 100)}% match
      </button>
      {open && <div className="source-tooltip">{source.chunk_text}</div>}
    </div>
  );
}

function Message({ msg }) {
  return (
    <div className={`msg ${msg.role}`}>
      <div className="bubble">{msg.content}</div>
      {msg.sources?.length > 0 && (
        <div className="sources">
          {msg.sources.map((s, i) => <SourceChip key={i} source={s} />)}
        </div>
      )}
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [docs, setDocs] = useState([]);
  const [activeDoc, setActiveDoc] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [drag, setDrag] = useState(false);
  const [toast, setToast] = useState(null);
  const [expandedSource, setExpandedSource] = useState(null);
  const chatRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => { fetchDocs(); }, []);
  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages, loading]);

  function showToast(msg) {
    setToast(msg);
    setTimeout(() => setToast(null), 3500);
  }

  async function fetchDocs() {
    try {
      const res = await fetch(`${API}/documents`);
      const data = await res.json();
      setDocs(data.documents);
    } catch { showToast("Could not reach the API. Is the backend running?"); }
  }

  async function uploadFile(file) {
    if (!file || !file.name.endsWith(".pdf")) return showToast("Please upload a PDF file.");
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error();
      const data = await res.json();
      await fetchDocs();
      setActiveDoc({ doc_id: data.doc_id, filename: data.filename });
      setMessages([]);
    } catch { showToast("Upload failed. Check the backend logs."); }
    finally { setUploading(false); }
  }

  async function sendMessage() {
    const q = input.trim();
    if (!q || loading) return;
    if (!activeDoc) return showToast("Select or upload a document first.");

    const userMsg = { role: "user", content: q };
    setMessages(m => [...m, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, doc_id: activeDoc.doc_id }),
      });
      const data = await res.json();
      setMessages(m => [...m, { role: "assistant", content: data.answer, sources: data.sources }]);
    } catch { showToast("Query failed. Check the backend."); }
    finally { setLoading(false); }
  }

  async function deleteDoc(doc_id, e) {
    e.stopPropagation();
    await fetch(`${API}/documents/${doc_id}`, { method: "DELETE" });
    if (activeDoc?.doc_id === doc_id) { setActiveDoc(null); setMessages([]); }
    await fetchDocs();
  }

  function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  }

  function handleTextarea(e) {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px";
  }

  return (
    <>
      <style>{css}</style>
      <div className="app">

        {/* ── Sidebar ── */}
        <aside className="sidebar">
          <div className="logo">PDF<span>/</span>QA</div>

          <label
            className={`upload-zone ${drag ? "drag" : ""}`}
            onDragOver={e => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={e => { e.preventDefault(); setDrag(false); uploadFile(e.dataTransfer.files[0]); }}
          >
            <input type="file" accept=".pdf" onChange={e => uploadFile(e.target.files[0])} />
            <strong>{uploading ? "Indexing…" : "Upload PDF"}</strong>
            <p>Drop a file or click to browse</p>
          </label>

          <div className="doc-list">
            <div className="doc-list-label">DOCUMENTS</div>
            {docs.length === 0 && <p style={{ fontSize: 12, color: "var(--muted)" }}>No documents yet.</p>}
            {docs.map(doc => (
              <div
                key={doc.doc_id}
                className={`doc-item ${activeDoc?.doc_id === doc.doc_id ? "active" : ""}`}
                onClick={() => { setActiveDoc(doc); setMessages([]); }}
              >
                <span className="doc-name">📄 {doc.filename}</span>
                <span className="del" onClick={e => deleteDoc(doc.doc_id, e)}>×</span>
              </div>
            ))}
          </div>
        </aside>

        {/* ── Main ── */}
        <main className="main">
          <div className="topbar">
            <span className="topbar-title">ASK YOUR DOCUMENT</span>
            <span className="topbar-doc">{activeDoc ? activeDoc.filename : "—"}</span>
          </div>

          <div className="chat" ref={chatRef}>
            {messages.length === 0 && !loading && (
              <div className="empty-state">
                <div className="icon">🔍</div>
                <p>{activeDoc ? `Ask anything about "${activeDoc.filename}"` : "Upload a PDF from the sidebar to get started."}</p>
              </div>
            )}
            {messages.map((msg, i) => <Message key={i} msg={msg} />)}
            {loading && (
              <div className="msg assistant">
                <div className="thinking">
                  <div className="dot" /><div className="dot" /><div className="dot" />
                </div>
              </div>
            )}
          </div>

          <div className="inputbar">
            <textarea
              ref={textareaRef}
              rows={1}
              placeholder={activeDoc ? "Ask a question about the document…" : "Upload a document to start asking questions"}
              value={input}
              onChange={handleTextarea}
              onKeyDown={handleKey}
              disabled={!activeDoc || loading}
            />
            <button className="send-btn" onClick={sendMessage} disabled={!activeDoc || loading || !input.trim()}>
              SEND ↑
            </button>
          </div>
        </main>
      </div>

      {toast && <div className="toast">{toast}</div>}
    </>
  );
}
