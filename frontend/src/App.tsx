
import "./App.css";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  BrainCircuit,
  Plus,
  MessageSquare,
  FileText,
  Upload,
  Send,
  Paperclip,
  Sparkles,
  Trash2,
  BookOpen,
  ChevronDown,
  CheckCircle2,
  LoaderCircle,
  Bot,
  User,
  PanelLeftClose,
  PanelLeftOpen,
  AlertCircle,
  RefreshCw,
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000";

interface DocumentItem {
  filename: string;
  chunks: number;
}

interface SourceItem {
  source: string;
  page: number;
}

interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources?: SourceItem[];
}

function App() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selectedDocument, setSelectedDocument] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [error, setError] = useState("");
  const [backendOnline, setBackendOnline] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load documents from backend
  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_URL}/api/documents`);

      if (!response.ok) {
        throw new Error("Could not load documents");
      }

      const data = await response.json();

      setDocuments(data.documents || []);
      setBackendOnline(true);

      setSelectedDocument((current) => {
        if (current && data.documents?.some(
          (doc: DocumentItem) => doc.filename === current
        )) {
          return current;
        }

        return data.documents?.[0]?.filename || "";
      });
    } catch {
      setBackendOnline(false);
      setError("Cannot connect to the backend. Make sure FastAPI is running.");
    }
  };

  // Check backend when app loads
  useEffect(() => {
    fetchDocuments();
  }, []);

  // Scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  // Upload PDF
  const handleUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Please upload a PDF file.");
      return;
    }

    if (file.size > 20 * 1024 * 1024) {
      setError("File size must be less than 20 MB.");
      return;
    }

    setUploading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "PDF upload failed.");
      }

      await fetchDocuments();

      setSelectedDocument(data.filename || file.name);

      setMessages((previous) => [
        ...previous,
        {
          id: Date.now(),
          role: "assistant",
          content: `**${file.name}** has been uploaded successfully. You can now ask questions about this document.`,
          sources: [],
        },
      ]);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Something went wrong during upload."
      );
    } finally {
      setUploading(false);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  // File input handler
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];

    if (file) {
      handleUpload(file);
    }
  };

  // Drag and drop handlers
  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);

    const file = event.dataTransfer.files?.[0];

    if (file) {
      handleUpload(file);
    }
  };

  // Ask a question
  const handleSend = async (customQuestion?: string) => {
    const questionText = (customQuestion ?? question).trim();

    if (!questionText || loading) return;

    if (!backendOnline) {
      setError("Backend is offline. Start the FastAPI server first.");
      return;
    }

    setError("");
    setQuestion("");

    const userMessage: ChatMessage = {
      id: Date.now(),
      role: "user",
      content: questionText,
    };

    setMessages((previous) => [...previous, userMessage]);
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: questionText,
          source: selectedDocument || null,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to get an answer.");
      }

      const assistantMessage: ChatMessage = {
        id: Date.now() + 1,
        role: "assistant",
        content: data.answer || "I couldn't generate an answer.",
        sources: data.sources || [],
      };

      setMessages((previous) => [...previous, assistantMessage]);
    } catch (err) {
      setMessages((previous) => [
        ...previous,
        {
          id: Date.now() + 1,
          role: "assistant",
          content:
            err instanceof Error
              ? `Sorry, something went wrong: ${err.message}`
              : "Sorry, something went wrong.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Start a new chat
  const handleNewChat = () => {
    setMessages([]);
    setQuestion("");
    setError("");
  };

  // Remove document from current selection
  const handleRemoveDocument = (filename: string) => {
    setDocuments((previous) =>
      previous.filter((doc) => doc.filename !== filename)
    );

    if (selectedDocument === filename) {
      const remaining = documents.filter(
        (doc) => doc.filename !== filename
      );

      setSelectedDocument(remaining[0]?.filename || "");
    }

    setError(
      "This only changes the current selection. It does not delete the PDF from the backend."
    );
  };

  const suggestions = [
    "Summarize this document",
    "Explain the important topics",
    "Give me exam questions",
    "Explain with simple examples",
  ];

  return (
    <div className="app-shell">
      {/* Sidebar */}
      {sidebarOpen && (
        <aside className="sidebar">
          <div className="brand">
            <div className="brand-icon">
              <BrainCircuit size={23} />
            </div>

            <div className="brand-text">
              <h2>DocuMind<span> AI</span></h2>
              <p>Intelligent document assistant</p>
            </div>
          </div>

          <button className="new-chat" onClick={handleNewChat}>
            <Plus size={17} />
            <span>New conversation</span>
          </button>

          <div className="side-section">
            <p className="side-label">WORKSPACE</p>

            <div className="nav-item active">
              <MessageSquare size={17} />
              <span>AI Chat</span>
            </div>

            <div className="nav-item">
              <BookOpen size={17} />
              <span>My Documents</span>
              <span className="nav-count">{documents.length}</span>
            </div>
          </div>

          <div className="side-section recent-section">
            <p className="side-label">RECENT ACTIVITY</p>

            {messages.length > 0 ? (
              <div className="recent-chat">
                <MessageSquare size={15} />
                <span>
                  {messages.find((message) => message.role === "user")
                    ?.content.slice(0, 24) || "Current conversation"}
                </span>
              </div>
            ) : (
              <p className="empty-recent">No recent conversations</p>
            )}
          </div>

          <div className="sidebar-bottom">
            <div className="plan-card">
              <div className="plan-icon">
                <Sparkles size={17} />
              </div>
              <div>
                <strong>Powered by AI</strong>
                <p>Hybrid RAG technology</p>
              </div>
            </div>

            <div className="profile">
              <div className="avatar">D</div>
              <div className="profile-info">
                <strong>DocuMind User</strong>
                <span>Personal workspace</span>
              </div>
            </div>
          </div>
        </aside>
      )}

      {/* Main Panel */}
      <main className="main-panel">
        {/* Topbar */}
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="icon-button"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              title="Toggle sidebar"
            >
              {sidebarOpen ? (
                <PanelLeftClose size={19} />
              ) : (
                <PanelLeftOpen size={19} />
              )}
            </button>

            <span className="slash">/</span>
            <span>Workspace</span>
            <span className="slash">/</span>
            <strong>AI Chat</strong>
          </div>

          <div className="topbar-right">
            <div className="connection-status">
              <span
                className={`status-dot ${backendOnline ? "online" : "offline"}`}
              />
              {backendOnline ? "Backend connected" : "Backend offline"}
            </div>

            <button
              className="icon-button"
              onClick={fetchDocuments}
              title="Refresh documents"
            >
              <RefreshCw size={17} />
            </button>
          </div>
        </header>

        {/* Main Content */}
        <div className="content">
          {/* Welcome Section */}
          <section className="welcome">
            <div className="welcome-icon">
              <BrainCircuit size={27} />
            </div>

            <p className="eyebrow">
              <Sparkles size={13} />
              YOUR PERSONAL AI KNOWLEDGE ASSISTANT
            </p>

            <h1>
              Understand your documents.
              <br />
              <span>Ask anything.</span>
            </h1>

            <p className="subtitle">
              Upload your PDFs and get accurate, context-aware answers
              powered by Hybrid Retrieval-Augmented Generation.
            </p>
          </section>

          {/* Upload Card */}
          <section
            className={`upload-card ${dragging ? "dragging" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              hidden
              onChange={handleFileChange}
            />

            <div className="upload-icon">
              {uploading ? (
                <LoaderCircle className="spin" size={25} />
              ) : (
                <Upload size={25} />
              )}
            </div>

            <div className="upload-text">
              <h3>
                {uploading ? "Processing your PDF..." : "Upload your documents"}
              </h3>

              <p>
                Drag and drop your PDF here, or{" "}
                <span>browse files</span>
              </p>

              <small>PDF files only · Maximum file size 20 MB</small>
            </div>

            <button
              className="browse-button"
              disabled={uploading}
              onClick={(event) => {
                event.stopPropagation();
                fileInputRef.current?.click();
              }}
            >
              <Paperclip size={15} />
              Browse
            </button>
          </section>

          {/* Documents Card */}
          <section className="documents-card">
            <div className="section-heading">
              <div>
                <h3>
                  <FileText size={17} />
                  Your documents
                </h3>
                <p>Select a document to ask questions about it.</p>
              </div>

              <span className="document-count">
                {documents.length} files
              </span>
            </div>

            {documents.length > 0 ? (
              <div className="file-list">
                {documents.map((doc) => (
                  <div className="file-row" key={doc.filename}>
                    <div className="pdf-icon">
                      <FileText size={19} />
                    </div>

                    <div className="file-info">
                      <strong title={doc.filename}>{doc.filename}</strong>
                      <span>{doc.chunks} indexed chunks</span>
                    </div>

                    {selectedDocument === doc.filename && (
                      <CheckCircle2
                        className="selected-check"
                        size={17}
                      />
                    )}

                    <button
                      className="remove-button"
                      title="Remove from current selection"
                      onClick={() => handleRemoveDocument(doc.filename)}
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="no-documents">
                <FileText size={23} />
                <p>No documents found. Upload a PDF to get started.</p>
              </div>
            )}
          </section>

          {/* Document Selector */}
          <section className="document-selector-card">
            <div className="selector-heading">
              <div>
                <h3>
                  <BookOpen size={17} />
                  Knowledge source
                </h3>
                <p>Choose which PDF DocuMind should search.</p>
              </div>

              <span className="selector-label">ACTIVE SOURCE</span>
            </div>

            <div className="select-wrapper">
              <select
                value={selectedDocument}
                onChange={(event) => setSelectedDocument(event.target.value)}
                disabled={documents.length === 0}
              >
                {documents.length === 0 && (
                  <option value="">No documents available</option>
                )}

                {documents.map((doc) => (
                  <option key={doc.filename} value={doc.filename}>
                    {doc.filename}
                  </option>
                ))}
              </select>

              <ChevronDown size={17} />
            </div>

            <div className="selected-document-info">
              <CheckCircle2 size={14} />
              {selectedDocument
                ? `Questions will be answered using ${selectedDocument}`
                : "Upload a PDF to select a knowledge source."}
            </div>
          </section>

          {/* Error Message */}
          {error && (
            <div className="error-message">
              <AlertCircle size={17} />
              <span>{error}</span>
              <button onClick={() => setError("")}>Dismiss</button>
            </div>
          )}

          {/* Chat Area */}
          <section className="chat-area">
            <div className="chat-heading">
              <div>
                <h3>
                  <Sparkles size={17} />
                  Ask DocuMind
                </h3>
                <p>Your AI-powered document conversation</p>
              </div>

              <button className="clear-chat" onClick={handleNewChat}>
                <Plus size={15} />
                New chat
              </button>
            </div>

            {/* Empty State */}
            {messages.length === 0 && (
              <div className="empty-chat">
                <div className="empty-icon">
                  <Bot size={29} />
                </div>

                <h3>What would you like to know?</h3>

                <p>
                  Ask a question about your selected document.
                  DocuMind will find relevant information and generate an answer.
                </p>

                <div className="suggestions">
                  {suggestions.map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => handleSend(suggestion)}
                      disabled={!selectedDocument || loading}
                    >
                      <Sparkles size={13} />
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Messages */}
            {messages.length > 0 && (
              <div className="messages">
                {messages.map((message) => (
                  <div
                    className={`message ${message.role}`}
                    key={message.id}
                  >
                    <div className="message-avatar">
                      {message.role === "assistant" ? (
                        <BrainCircuit size={18} />
                      ) : (
                        <User size={18} />
                      )}
                    </div>

                    <div className="message-content">
                      <div className="message-name">
                        {message.role === "assistant" ? "DocuMind AI" : "You"}
                      </div>

                      <div className="markdown-content">
                        <ReactMarkdown>
                          {message.content}
                        </ReactMarkdown>
                      </div>

                      {message.sources && message.sources.length > 0 && (
                        <div className="source-list">
                          <div className="source-heading">
                            <BookOpen size={14} />
                            Sources
                          </div>

                          {message.sources.map((source, index) => (
                            <div
                              className="source-item"
                              key={`${source.source}-${source.page}-${index}`}
                            >
                              <FileText size={14} />
                              <span>{source.source}</span>
                              <span className="source-page">
                                Page {source.page}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {loading && (
                  <div className="message assistant">
                    <div className="message-avatar">
                      <BrainCircuit size={18} />
                    </div>

                    <div className="message-content">
                      <div className="message-name">DocuMind AI</div>
                      <div className="typing-indicator">
                        <LoaderCircle className="spin" size={16} />
                        Searching documents and generating an answer...
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}

            {/* Chat Input */}
            <div className="chat-input-wrap">
              <form
                className="chat-input"
                onSubmit={(event) => {
                  event.preventDefault();
                  handleSend();
                }}
              >
                <input
                  type="text"
                  placeholder={
                    selectedDocument
                      ? "Ask anything about your document..."
                      : "Upload and select a PDF first..."
                  }
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  disabled={loading || !selectedDocument}
                />

                <button
                  type="submit"
                  className="send-button"
                  disabled={!question.trim() || loading || !selectedDocument}
                  title="Send message"
                >
                  {loading ? (
                    <LoaderCircle className="spin" size={18} />
                  ) : (
                    <Send size={18} />
                  )}
                </button>
              </form>

              <div className="input-footer">
                <span>
                  <Sparkles size={12} />
                  Powered by Gemini + Hybrid RAG
                </span>

                <span>DocuMind AI may make mistakes. Verify important information.</span>
              </div>
            </div>
          </section>

          <footer className="app-footer">
            <span>DocuMind AI</span>
            <span>Built with React, FastAPI, ChromaDB and Gemini</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;

