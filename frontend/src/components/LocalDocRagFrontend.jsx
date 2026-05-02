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
  Trash2,
  ShieldAlert,
  Bot,
  Search,
  TriangleAlert,
  FileUp,
  CircleHelp,
} from "lucide-react";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ||
  "http://127.0.0.1:8000";

const acceptedTypes = ".pdf,.docx,.txt";

const statusConfig = {
  uploaded: {
    label: "已上传",
    className: "status uploaded",
    icon: FileUp,
    retryable: false,
  },
  indexing: {
    label: "索引中",
    className: "status indexing",
    icon: Loader2,
    retryable: false,
  },
  indexed: {
    label: "已索引",
    className: "status indexed",
    icon: CheckCircle2,
    retryable: false,
  },
  upload_failed: {
    label: "上传失败",
    className: "status failed",
    icon: AlertCircle,
    retryable: false,
  },
  index_failed: {
    label: "索引失败",
    className: "status failed",
    icon: AlertCircle,
    retryable: true,
  },
};

function safeJsonParse(text, fallback) {
  try {
    return JSON.parse(text);
  } catch {
    return fallback;
  }
}

function makeSessionId() {
  const key = "rrag_session_id";
  const old = localStorage.getItem(key);
  if (old) return old;
  const created = `session-${Date.now()}`;
  localStorage.setItem(key, created);
  return created;
}

function RiskBadge({ risk }) {
  if (!risk) return null;

  const level = risk.risk_level || "unknown";
  const map = {
    low: "risk-badge low",
    medium: "risk-badge medium",
    high: "risk-badge high",
    unknown: "risk-badge unknown",
  };

  return (
    <div className={map[level] || map.unknown}>
      <ShieldAlert size={14} />
      <span>
        风险：{level}（{Number(risk.risk_score ?? 0).toFixed(2)}）
      </span>
    </div>
  );
}

function ReflectionPanel({ reflections }) {
  if (!reflections?.length) return null;

  return (
    <div className="meta-panel">
      <div className="meta-title">Self-RAG 反思信息</div>
      {reflections.map((item, idx) => (
        <div className="reflection-card" key={`${idx}-${item.round}`}>
          <div className="reflection-grid">
            <div><strong>轮次：</strong>{item.round}</div>
            <div><strong>Retrieve：</strong>{item.retrieve}</div>
            <div><strong>相关性：</strong>{item.isrel}</div>
            <div><strong>支持度：</strong>{item.issup}</div>
            <div><strong>有用性：</strong>{item.isuse}</div>
            <div><strong>FollowUp：</strong>{item.followup_query || "None"}</div>
          </div>
          {item.raw_text ? (
            <details className="raw-box">
              <summary>查看原始模型输出</summary>
              <pre>{item.raw_text}</pre>
            </details>
          ) : null}
        </div>
      ))}
    </div>
  );
}

function TracePanel({ trace }) {
  if (!trace?.length) return null;

  return (
    <div className="meta-panel">
      <div className="meta-title">执行轨迹</div>
      <div className="trace-list">
        {trace.map((item, idx) => (
          <div className="trace-item" key={`${item.stage}-${idx}`}>
            <div className="trace-head">
              <span className="trace-stage">{item.stage}</span>
              {item.round ? <span className="trace-round">Round {item.round}</span> : null}
            </div>
            {item.query ? <div className="trace-line"><strong>Query：</strong>{item.query}</div> : null}
            {item.retrieve ? <div className="trace-line"><strong>Retrieve：</strong>{item.retrieve}</div> : null}
            {typeof item.retrieved_count === "number" ? (
              <div className="trace-line"><strong>Retrieved：</strong>{item.retrieved_count}</div>
            ) : null}
            {item.detail ? <div className="trace-line"><strong>Detail：</strong>{item.detail}</div> : null}
            {item.top_sources?.length ? (
              <div className="trace-line">
                <strong>Top Sources：</strong>
                <ul className="trace-sources">
                  {item.top_sources.map((src, i) => (
                    <li key={`${src}-${i}`}>{src}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
}

function CandidatesPanel({ candidates }) {
  if (!candidates?.length) return null;

  return (
    <div className="meta-panel">
      <div className="meta-title">候选答案</div>
      <div className="candidate-list">
        {candidates.map((item, idx) => (
          <div className="candidate-card" key={idx}>
            <div className="candidate-score-row">
              <span>候选 {idx + 1}</span>
              <span>final_score = {Number(item.final_score ?? 0).toFixed(4)}</span>
            </div>
            <div className="candidate-score-sub">
              evidence_score = {Number(item.evidence_score ?? 0).toFixed(4)}，
              retrieve = {item.retrieve}，
              isrel = {item.isrel}，
              issup = {item.issup}，
              isuse = {item.isuse}
              {item.subquery ? `，subquery = ${item.subquery}` : ""}
            </div>
            <div className="candidate-answer">{item.answer || "无内容"}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function CitationsPanel({ citations }) {
  if (!citations?.length) return null;

  return (
    <div className="meta-panel">
      <div className="meta-title">参考来源</div>
      <div className="citation-list">
        {citations.map((c) => (
          <div className="citation-card" key={`${c.index}-${c.source}`}>
            <div className="citation-source">
              [{c.index}] {c.source}
            </div>
            <div className="citation-excerpt">{c.excerpt}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SubAnswersPanel({ subAnswers }) {
  if (!subAnswers?.length) return null;

  return (
    <div className="meta-panel">
      <div className="meta-title">子查询结果</div>
      <div className="subanswer-list">
        {subAnswers.map((item, idx) => (
          <div className="subanswer-card" key={`${idx}-${item.subquery}`}>
            <div className="subanswer-head">
              <span className="subanswer-index">子查询 {idx + 1}</span>
              <span className="subanswer-score">
                best_score = {Number(item.best_score ?? 0).toFixed(4)}
              </span>
            </div>

            <div className="subanswer-block">
              <div className="subanswer-label">子查询</div>
              <div className="subanswer-query">{item.subquery || "无"}</div>
            </div>

            <div className="subanswer-block">
              <div className="subanswer-label">子查询答案</div>
              <div className="subanswer-answer">{item.answer || "无内容"}</div>
            </div>

            <div className="subanswer-block">
              <div className="subanswer-label">对应证据块</div>
              {item.citations?.length ? (
                <div className="citation-list compact">
                  {item.citations.map((c) => (
                    <div className="citation-card" key={`${idx}-${c.index}-${c.source}`}>
                      <div className="citation-source">
                        [{c.index}] {c.source}
                      </div>
                      <div className="citation-excerpt">{c.excerpt}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="subanswer-empty">无证据块</div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function AssistantMessage({ msg }) {
  const showSubAnswers =
    (msg.query_mode === "fuzzy" || msg.query_mode === "complex") &&
    msg.sub_answers?.length > 0;

  return (
    <div className="message assistant">
      <div className="avatar"><Bot size={18} /></div>
      <div className="bubble-wrap">
        <div className="bubble">{msg.content || "未返回答案。"}</div>

        {showSubAnswers ? (
          <SubAnswersPanel subAnswers={msg.sub_answers} />
        ) : (
          <>
            <CitationsPanel citations={msg.citations} />
            <CandidatesPanel candidates={msg.candidates} />
          </>
        )}

        <ReflectionPanel reflections={msg.reflections} />
        <TracePanel trace={msg.trace} />

        <div className="risk-row">
          <RiskBadge risk={msg.retrieval_risk} />
          <RiskBadge risk={msg.generation_risk} />
        </div>
      </div>
    </div>
  );
}

function UserMessage({ msg }) {
  return (
    <div className="message user">
      <div className="bubble-wrap">
        <div className="bubble">{msg.content}</div>
      </div>
      <div className="avatar user"><MessageSquare size={18} /></div>
    </div>
  );
}

export default function LocalDocRagFrontend() {
  const [files, setFiles] = useState([]);
  const [query, setQuery] = useState("");
  const [dragging, setDragging] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [globalError, setGlobalError] = useState("");
  const [sessionId] = useState(() => makeSessionId());

  const fileInputRef = useRef(null);
  const chatRef = useRef(null);

  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content:
        "你好，请先上传 PDF、DOCX 或 TXT 文档。文件完成索引后，就可以基于当前文档进行问答。",
      query_mode: "simple",
      citations: [],
      reflections: [],
      trace: [],
      candidates: [],
      sub_answers: [],
      retrieval_risk: null,
      generation_risk: null,
    },
  ]);

  const indexedCount = useMemo(
    () => files.filter((f) => f.status === "indexed").length,
    [files]
  );

  const hasIndexing = useMemo(
    () => files.some((f) => f.status === "indexing"),
    [files]
  );

  const fetchFiles = async () => {
    try {
      setLoadingFiles(true);
      const res = await fetch(`${API_BASE}/files`);
      const data = await res.json();
      setFiles(data.files || []);
      setGlobalError("");
    } catch (e) {
      console.error("获取文件列表失败", e);
      setGlobalError("无法连接后端或获取文件列表失败。");
    } finally {
      setLoadingFiles(false);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  useEffect(() => {
    if (!hasIndexing) return;
    const timer = setInterval(() => {
      fetchFiles();
    }, 2500);
    return () => clearInterval(timer);
  }, [hasIndexing]);

  useEffect(() => {
    if (!chatRef.current) return;
    chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages, submitting]);

  const uploadFiles = async (incomingFiles) => {
    const list = Array.from(incomingFiles || []);
    if (!list.length) return;

    const formData = new FormData();
    list.forEach((file) => formData.append("files", file));

    setUploading(true);
    setGlobalError("");

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      const text = await res.text();
      const data = safeJsonParse(text, {});
      if (!res.ok) {
        throw new Error(data.detail || "上传失败");
      }

      setFiles(data.files || []);
    } catch (e) {
      console.error("上传失败", e);
      setGlobalError(e.message || "上传失败");
    } finally {
      setUploading(false);
    }
  };

  const onFilesSelected = (event) => {
    const selected = event.target.files;
    if (selected?.length) uploadFiles(selected);
    event.target.value = "";
  };

  const onDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files;
    if (dropped?.length) uploadFiles(dropped);
  };

  const retryIndex = async (filename) => {
    try {
      const res = await fetch(
        `${API_BASE}/files/${encodeURIComponent(filename)}/retry-index`,
        { method: "POST" }
      );
      const text = await res.text();
      const data = safeJsonParse(text, {});
      if (!res.ok) throw new Error(data.detail || "重试索引失败");
      fetchFiles();
    } catch (e) {
      console.error("重试索引失败", e);
      setGlobalError(e.message || "重试索引失败");
    }
  };

  const removeFile = async (filename) => {
    try {
      const res = await fetch(
        `${API_BASE}/files/${encodeURIComponent(filename)}`,
        { method: "DELETE" }
      );
      const text = await res.text();
      const data = safeJsonParse(text, {});
      if (!res.ok) throw new Error(data.detail || "删除文件失败");
      setFiles(data.files || []);
    } catch (e) {
      console.error("删除文件失败", e);
      setGlobalError(e.message || "删除文件失败");
    }
  };

  const handleSend = async () => {
    const trimmed = query.trim();
    if (!trimmed || submitting) return;

    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: trimmed,
    };

    setMessages((prev) => [...prev, userMessage]);
    setQuery("");
    setSubmitting(true);
    setGlobalError("");

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: trimmed,
          session_id: sessionId,
        }),
      });

      const text = await res.text();
      const data = safeJsonParse(text, {});
      if (!res.ok) {
        throw new Error(data.detail || "提问失败");
      }

      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: data.answer || "未返回答案。",
        query_mode: data.query_mode || "simple",
        citations: data.citations || [],
        reflections: data.reflections || [],
        trace: data.trace || [],
        candidates: data.candidates || [],
        sub_answers: data.sub_answers || [],
        retrieval_risk: data.retrieval_risk || null,
        generation_risk: data.generation_risk || null,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          id: `assistant-error-${Date.now()}`,
          role: "assistant",
          content: `请求失败：${e.message}`,
          query_mode: "simple",
          citations: [],
          reflections: [],
          trace: [],
          candidates: [],
          sub_answers: [],
          retrieval_risk: null,
          generation_risk: null,
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

  return (
    <div className="page">
      <div className="page-inner">
        <header className="hero">
          <div>
            <h1>文档智能问答系统</h1>
            <p>
              便捷的文档智能检索服务
            </p>
          </div>
          <div className="hero-stats">
            <div className="stat-card">
              <span>已索引文件</span>
              <strong>{indexedCount}</strong>
            </div>
            <div className="stat-card">
              <span>文件总数</span>
              <strong>{files.length}</strong>
            </div>
            <div className="stat-card">
              <span>会话 ID</span>
              <strong className="small">{sessionId}</strong>
            </div>
          </div>
        </header>

        {globalError ? (
          <div className="global-alert error">
            <TriangleAlert size={16} />
            <span>{globalError}</span>
          </div>
        ) : null}

        <div className="layout">
          <aside className="sidebar">
            <div className="sidebar-top">
              <section className="panel side-equal-panel">
                <div className="panel-head upload-panel-head">
                  <h2>文件上传</h2>
                  <span className="upload-format-inline">支持格式：PDF、DOCX、TXT</span>
                </div>

                <div
                  className={`upload-box ${dragging ? "dragging" : ""}`}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragging(true);
                  }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={onDrop}
                >
                  <div className="upload-box-fill">
                    <Upload size={20} />
                    <div className="upload-title">拖拽文件到这里或点击下方按钮上传</div>
                    <button
                      className="primary-btn upload-btn"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={uploading}
                    >
                      {uploading ? (
                        <>
                          <Loader2 size={16} className="spin" />
                          上传中...
                        </>
                      ) : (
                        <>
                          <Upload size={16} />
                          选择文件
                        </>
                      )}
                    </button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept={acceptedTypes}
                      multiple
                      hidden
                      onChange={onFilesSelected}
                    />
                  </div>
                </div>
              </section>

              <section className="panel side-equal-panel">
                <div className="panel-head">
                  <h2>文件列表</h2>
                  <button className="ghost-btn" onClick={fetchFiles} disabled={loadingFiles}>
                    <RefreshCw size={16} className={loadingFiles ? "spin" : ""} />
                    刷新
                  </button>
                </div>

                <div className="file-scroll-area">
                  {files.length === 0 ? (
                    <div className="empty-box">
                      <FileText size={18} />
                      <span>暂无文件，请先上传文档。</span>
                    </div>
                  ) : (
                    <div className="file-list">
                      {files.map((file) => {
                        const conf = statusConfig[file.status] || statusConfig.upload_failed;
                        const Icon = conf.icon || CircleHelp;

                        return (
                          <div className="file-card" key={file.name}>
                            <div className="file-main">
                              <div className="file-name-row">
                                <FileText size={16} />
                                <span className="file-name">{file.name}</span>
                              </div>

                              <div className="file-side-actions">
                                <div className={`${conf.className} file-status-pill`}>
                                  <Icon
                                    size={14}
                                    className={file.status === "indexing" ? "spin" : ""}
                                  />
                                  <span>{conf.label}</span>
                                </div>

                                {conf.retryable ? (
                                  <button
                                    className="small-btn file-action-btn"
                                    onClick={() => retryIndex(file.name)}
                                  >
                                    <RefreshCw size={14} />
                                    重试索引
                                  </button>
                                ) : (
                                  <button
                                    className="small-btn danger file-action-btn"
                                    onClick={() => removeFile(file.name)}
                                  >
                                    <Trash2 size={14} />
                                    删除
                                  </button>
                                )}
                              </div>
                            </div>

                            {file.error ? <div className="file-error">{file.error}</div> : null}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </section>
            </div>
          </aside>

          <main className="main">
            <section className="panel chat-panel">
              <div className="panel-head">
                <h2>问答页面</h2>
                <span className="panel-sub">
                  请在上传文档后向小助手提问吧！
                </span>
              </div>

              <div className="chat-window" ref={chatRef}>
                {messages.map((msg) =>
                  msg.role === "assistant" ? (
                    <AssistantMessage key={msg.id} msg={msg} />
                  ) : (
                    <UserMessage key={msg.id} msg={msg} />
                  )
                )}

                {submitting ? (
                  <div className="message assistant">
                    <div className="avatar"><Bot size={18} /></div>
                    <div className="bubble-wrap">
                      <div className="bubble loading-bubble">
                        <Loader2 size={16} className="spin" />
                        正在生成回答...
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="composer">
                <div className="composer-tip">
                  <Search size={14} />
                  <span>
                    {indexedCount > 0
                      ? "请输入你的问题，例如：这份文档的核心结论是什么？"
                      : "请先上传并索引文档，再开始提问"}
                  </span>
                </div>

                <div className="composer-row">
                  <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={onKeyDown}
                    rows={4}
                    placeholder={
                      indexedCount > 0
                        ? "请输入你的问题..."
                        : "请先上传并索引文档"
                    }
                    disabled={indexedCount === 0 || submitting}
                  />
                  <button
                    className="send-btn"
                    onClick={handleSend}
                    disabled={submitting || !query.trim() || indexedCount === 0}
                  >
                    {submitting ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
                    <span>发送</span>
                  </button>
                </div>
              </div>
            </section>
          </main>
        </div>
      </div>
    </div>
  );
}