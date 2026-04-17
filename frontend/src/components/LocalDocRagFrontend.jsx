import { useEffect, useMemo, useRef, useState } from "react";
import {
  Upload,
  FileText,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  MessageSquare,
  Send,
  Loader2,
} from "lucide-react";

export default function LocalDocRagFrontend() {
  const API_BASE = "http://127.0.0.1:8000";

  const [files, setFiles] = useState([]);
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content: "你好，请先上传 PDF、DOCX 或 TXT 文档。文件上传并完成索引后，你就可以开始提问。",
      citations: [],
    },
  ]);
  const [query, setQuery] = useState("");
  const [dragging, setDragging] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);

  const statusConfig = {
    uploaded: {
      label: "已上传",
      className: "bg-blue-50 text-blue-700 border-blue-200",
      icon: <CheckCircle2 className="w-4 h-4" />,
      retryable: false,
    },
    indexing: {
      label: "索引中",
      className: "bg-amber-50 text-amber-700 border-amber-200",
      icon: <Loader2 className="w-4 h-4 animate-spin" />,
      retryable: false,
    },
    indexed: {
      label: "已索引",
      className: "bg-emerald-50 text-emerald-700 border-emerald-200",
      icon: <CheckCircle2 className="w-4 h-4" />,
      retryable: false,
    },
    upload_failed: {
      label: "上传失败",
      className: "bg-red-50 text-red-700 border-red-200",
      icon: <AlertCircle className="w-4 h-4" />,
      retryable: false,
    },
    index_failed: {
      label: "索引失败",
      className: "bg-red-50 text-red-700 border-red-200",
      icon: <RefreshCw className="w-4 h-4" />,
      retryable: true,
    },
  };

  const indexedCount = useMemo(
    () => files.filter((f) => f.status === "indexed").length,
    [files]
  );

  const acceptedTypes = ".pdf,.docx,.txt";

  const fetchFiles = async () => {
    try {
      const res = await fetch(`${API_BASE}/files`);
      const data = await res.json();
      setFiles(data.files || []);
    } catch (e) {
      console.error("获取文件列表失败", e);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  const uploadFiles = async (incomingFiles) => {
    const list = Array.from(incomingFiles);
    if (list.length === 0) return;

    const formData = new FormData();
    list.forEach((file) => formData.append("files", file));

    setUploading(true);
    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setFiles(data.files || []);
    } catch (e) {
      console.error("上传失败", e);
    } finally {
      setUploading(false);
    }
  };

  const onFilesSelected = (event) => {
    const selected = event.target.files;
    if (selected && selected.length > 0) {
      uploadFiles(selected);
    }
    event.target.value = "";
  };

  const onDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files;
    if (dropped && dropped.length > 0) {
      uploadFiles(dropped);
    }
  };

  const retryIndex = async (filename) => {
    try {
      const res = await fetch(
        `${API_BASE}/files/${encodeURIComponent(filename)}/retry-index`,
        { method: "POST" }
      );
      await res.json();
      fetchFiles();
    } catch (e) {
      console.error("重试索引失败", e);
    }
  };

  const handleSend = async () => {
    const trimmed = query.trim();
    if (!trimmed || submitting) return;

    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: trimmed,
      citations: [],
    };

    setMessages((prev) => [...prev, userMessage]);
    setQuery("");
    setSubmitting(true);

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: trimmed }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "提问失败");
      }

      setMessages((prev) => [
        ...prev,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: data.answer || "未返回答案。",
          citations: data.citations || [],
        },
      ]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          id: `assistant-error-${Date.now()}`,
          role: "assistant",
          content: `请求失败：${e.message}`,
          citations: [],
        },
      ]);
    } finally {
      setSubmitting(false);
    }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const removeFile = async (filename) => {
  try {
    const res = await fetch(`${API_BASE}/files/${encodeURIComponent(filename)}`, {
      method: "DELETE",
    });
    const data = await res.json();
    setFiles(data.files || []);
  } catch (e) {
    console.error("删除文件失败", e);
  }
};

  return (
    <div style={{ minHeight: "100vh", background: "#f8fafc", padding: 24 }}>
      <div style={{ maxWidth: 1400, margin: "0 auto" }}>
        <div
          style={{
            background: "#fff",
            border: "1px solid #e2e8f0",
            borderRadius: 24,
            padding: 24,
            marginBottom: 24,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 16,
          }}
        >
          <div>
            <h1 style={{ margin: 0, fontSize: 28, color: "#0f172a" }}>
              本地文档智能问答系统
            </h1>
            <p style={{ marginTop: 8, color: "#64748b" }}>
              上传文档、查看索引状态，并基于已索引文件进行问答。
            </p>
          </div>
          <div
            style={{
              background: "#f1f5f9",
              borderRadius: 16,
              padding: "10px 16px",
              color: "#334155",
            }}
          >
            已索引文件 <b>{indexedCount}</b> / {files.length}
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 24,
          }}
        >
          <section
            style={{
              background: "#fff",
              border: "1px solid #e2e8f0",
              borderRadius: 24,
              padding: 20,
            }}
          >
            <div style={{ marginBottom: 16 }}>
              <h2 style={{ margin: 0, fontSize: 22, color: "#0f172a" }}>
                文件上传
              </h2>
              <p style={{ marginTop: 8, color: "#64748b" }}>
                支持 PDF、DOCX、TXT，可拖拽上传或手动选择文件。
              </p>
            </div>

            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              style={{
                border: dragging ? "2px dashed #38bdf8" : "2px dashed #cbd5e1",
                background: dragging ? "#f0f9ff" : "#f8fafc",
                borderRadius: 24,
                padding: 32,
                textAlign: "center",
              }}
            >
              <div style={{ marginBottom: 12 }}>
                <FileText size={36} color="#475569" />
              </div>
              <p style={{ margin: 0, fontSize: 18, color: "#1e293b" }}>
                拖动文件到此处上传
              </p>
              <p style={{ marginTop: 8, color: "#64748b" }}>
                或点击下方按钮选择文件
              </p>

              <button
                onClick={() => fileInputRef.current?.click()}
                style={{
                  marginTop: 20,
                  background: "#0f172a",
                  color: "#fff",
                  border: "none",
                  borderRadius: 14,
                  padding: "10px 16px",
                  cursor: "pointer",
                }}
              >
                {uploading ? "上传中..." : "选择文件上传"}
              </button>

              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept={acceptedTypes}
                onChange={onFilesSelected}
                style={{ display: "none" }}
              />
            </div>

            <div style={{ marginTop: 24 }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: 12,
                }}
              >
                <h3 style={{ margin: 0, color: "#1e293b" }}>已上传文件</h3>
                <span style={{ color: "#64748b" }}>共 {files.length} 个</span>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {files.length === 0 ? (
                  <div
                    style={{
                      border: "1px solid #e2e8f0",
                      borderRadius: 18,
                      background: "#f8fafc",
                      padding: 24,
                      textAlign: "center",
                      color: "#64748b",
                    }}
                  >
                    暂无文件，请先上传文档。
                  </div>
                ) : (
                  files.map((file) => {
                    const conf = statusConfig[file.status] || statusConfig.upload_failed;

                    return (
                      <div
                        key={file.name}
                        style={{
                          border: "1px solid #e2e8f0",
                          borderRadius: 18,
                          background: "#fff",
                          padding: 16,
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            gap: 16,
                            alignItems: "flex-start",
                          }}
                        >
                          <div style={{ flex: 1 }}>
                            <div style={{ fontWeight: 600, color: "#1e293b" }}>
                              {file.name}
                            </div>
                            {file.error ? (
                              <div style={{ marginTop: 8, color: "#dc2626", fontSize: 13 }}>
                                {file.error}
                              </div>
                            ) : null}
                          </div>

                          <button
                            onClick={() => conf.retryable && retryIndex(file.name)}
                            style={{
                              border: "1px solid #cbd5e1",
                              borderRadius: 999,
                              padding: "6px 12px",
                              cursor: conf.retryable ? "pointer" : "default",
                              background: "#fff",
                            }}
                          >
                            {conf.label}
                          </button>

                          <button
                            onClick={() => removeFile(file.name)}
                            style={{
                              border: "1px solid #cbd5e1",
                              borderRadius: 999,
                              padding: "6px 12px",
                              cursor: "pointer",
                              background: "#fff",
                              marginLeft: 8,
                            }}
                          >
                            移除文件
                        </button>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </section>

          <section
            style={{
              background: "#fff",
              border: "1px solid #e2e8f0",
              borderRadius: 24,
              padding: 20,
              display: "flex",
              flexDirection: "column",
              minHeight: 760,
            }}
          >
            <div style={{ marginBottom: 16 }}>
              <h2 style={{ margin: 0, fontSize: 22, color: "#0f172a" }}>
                问答页面
              </h2>
              <p style={{ marginTop: 8, color: "#64748b" }}>
                输入查询后，系统返回最终回答。
              </p>
            </div>

            <div
              style={{
                flex: 1,
                border: "1px solid #e2e8f0",
                borderRadius: 24,
                background: "#f8fafc",
                padding: 16,
                overflow: "auto",
                minHeight: 520,
              }}
            >
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  style={{
                    display: "flex",
                    justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
                    marginBottom: 16,
                  }}
                >
                  <div
                    style={{
                      maxWidth: "85%",
                      borderRadius: 20,
                      padding: 14,
                      background: msg.role === "user" ? "#0f172a" : "#fff",
                      color: msg.role === "user" ? "#fff" : "#1e293b",
                      border: msg.role === "user" ? "none" : "1px solid #e2e8f0",
                    }}
                  >
                    <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.8 }}>
                      {msg.content}
                    </div>

                    {msg.role === "assistant" && msg.citations?.length > 0 ? (
                      <div
                        style={{
                          marginTop: 12,
                          border: "1px solid #e2e8f0",
                          borderRadius: 16,
                          padding: 12,
                          background: "#f8fafc",
                          color: "#475569",
                          fontSize: 13,
                        }}
                      >
                        <div style={{ fontWeight: 600, marginBottom: 8 }}>参考来源</div>
                        {msg.citations.map((c) => (
                          <div key={`${msg.id}-${c.index}`} style={{ marginBottom: 8 }}>
                            <div style={{ fontWeight: 600 }}>
                              [{c.index}] {c.source}
                            </div>
                            <div>{c.excerpt}</div>
                          </div>
                        ))}
                      </div>
                    ) : null}
                  </div>
                </div>
              ))}

              {submitting ? (
                <div style={{ color: "#64748b" }}>正在生成回答...</div>
              ) : null}
            </div>

            <div
              style={{
                marginTop: 16,
                border: "1px solid #e2e8f0",
                borderRadius: 24,
                background: "#fff",
                padding: 12,
              }}
            >
              <div style={{ display: "flex", gap: 12, alignItems: "flex-end" }}>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={onKeyDown}
                  rows={3}
                  placeholder={
                    indexedCount > 0
                      ? "请输入你的查询，例如：这份文档的核心结论是什么？"
                      : "请先上传并索引文档，再开始提问"
                  }
                  style={{
                    flex: 1,
                    minHeight: 84,
                    resize: "none",
                    borderRadius: 16,
                    border: "1px solid #cbd5e1",
                    padding: 12,
                    fontSize: 14,
                  }}
                />
                <button
                  onClick={handleSend}
                  disabled={submitting || !query.trim()}
                  style={{
                    height: 84,
                    padding: "0 20px",
                    borderRadius: 16,
                    border: "none",
                    background: "#0f172a",
                    color: "#fff",
                    cursor: submitting || !query.trim() ? "not-allowed" : "pointer",
                    opacity: submitting || !query.trim() ? 0.5 : 1,
                  }}
                >
                  发送
                </button>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}