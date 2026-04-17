export default function LocalDocRagFrontend() {
  const { useEffect, useMemo, useRef, useState } = React;
  const {
    Upload,
    FileText,
    RefreshCw,
    CheckCircle2,
    AlertCircle,
    MessageSquare,
    Send,
    Loader2,
    Trash2,
  } = lucideReact;

  const API_BASE = "http://127.0.0.1:5173";

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

  const formatSize = (bytes) => {
    if (!bytes && bytes !== 0) return "--";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
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

  return (
    <div className="min-h-screen bg-slate-50 p-4 md:p-6">
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="flex items-center justify-between rounded-3xl bg-white px-6 py-5 shadow-sm border border-slate-200">
          <div>
            <h1 className="text-2xl font-semibold text-slate-900">本地文档智能问答系统</h1>
            <p className="mt-1 text-sm text-slate-500">上传文档、查看索引状态，并基于已索引文件进行问答。</p>
          </div>
          <div className="rounded-2xl bg-slate-100 px-4 py-2 text-sm text-slate-700">
            已索引文件 <span className="font-semibold">{indexedCount}</span> / {files.length}
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
          <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="mb-4 flex items-center gap-3">
              <div className="rounded-2xl bg-sky-100 p-2 text-sky-700">
                <Upload className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-slate-900">文件上传</h2>
                <p className="text-sm text-slate-500">支持 PDF、DOCX、TXT，可拖拽上传或手动选择文件。</p>
              </div>
            </div>

            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              className={`group rounded-3xl border-2 border-dashed p-8 text-center transition ${
                dragging
                  ? "border-sky-400 bg-sky-50"
                  : "border-slate-300 bg-slate-50 hover:border-sky-300 hover:bg-sky-50/40"
              }`}
            >
              <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-white shadow-sm border border-slate-200">
                <FileText className="w-7 h-7 text-slate-600" />
              </div>
              <p className="mt-4 text-base font-medium text-slate-800">拖动文件到此处上传</p>
              <p className="mt-1 text-sm text-slate-500">或点击下方按钮选择文件</p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="mt-5 inline-flex items-center gap-2 rounded-2xl bg-slate-900 px-4 py-2.5 text-sm font-medium text-white shadow-sm transition hover:bg-slate-800"
              >
                <Upload className="w-4 h-4" />
                {uploading ? "上传中..." : "选择文件上传"}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept={acceptedTypes}
                onChange={onFilesSelected}
                className="hidden"
              />
            </div>

            <div className="mt-6">
              <div className="mb-3 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-800">已上传文件</h3>
                <span className="text-xs text-slate-500">共 {files.length} 个</span>
              </div>

              <div className="space-y-3 max-h-[460px] overflow-auto pr-1">
                {files.length === 0 ? (
                  <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-8 text-center text-sm text-slate-500">
                    暂无文件，请先上传文档。
                  </div>
                ) : (
                  files.map((file) => {
                    const conf = statusConfig[file.status] || statusConfig.upload_failed;
                    return (
                      <div
                        key={file.name}
                        className="rounded-2xl border border-slate-200 bg-white px-4 py-4 shadow-sm"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-2">
                              <FileText className="w-4 h-4 text-slate-500 shrink-0" />
                              <p className="truncate font-medium text-slate-800">{file.name}</p>
                            </div>
                            <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500">
                              <span>{file.path?.split(".").pop()?.toUpperCase() || "--"}</span>
                            </div>
                            {file.error ? (
                              <p className="mt-2 text-xs text-red-600">{file.error}</p>
                            ) : null}
                          </div>

                          <div className="flex items-center gap-2 shrink-0">
                            <button
                              onClick={() => conf.retryable && retryIndex(file.name)}
                              className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium ${conf.className} ${
                                conf.retryable ? "hover:opacity-90 cursor-pointer" : "cursor-default"
                              }`}
                              title={conf.retryable ? "点击重新索引" : conf.label}
                            >
                              {conf.icon}
                              {conf.label}
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </section>

          <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm flex flex-col min-h-[760px]">
            <div className="mb-4 flex items-center gap-3">
              <div className="rounded-2xl bg-violet-100 p-2 text-violet-700">
                <MessageSquare className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-slate-900">问答页面</h2>
                <p className="text-sm text-slate-500">输入查询后，系统返回最终回答。</p>
              </div>
            </div>

            <div className="flex-1 rounded-3xl bg-slate-50 border border-slate-200 p-4 overflow-auto space-y-4 min-h-[520px]">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[85%] rounded-3xl px-4 py-3 shadow-sm ${
                      msg.role === "user"
                        ? "bg-slate-900 text-white"
                        : "bg-white text-slate-800 border border-slate-200"
                    }`}
                  >
                    <div className="text-sm leading-7 whitespace-pre-wrap">{msg.content}</div>
                    {msg.role === "assistant" && msg.citations?.length > 0 ? (
                      <div className="mt-3 rounded-2xl bg-slate-50 px-3 py-3 text-xs text-slate-600 border border-slate-200">
                        <div className="mb-2 font-medium text-slate-700">参考来源</div>
                        <div className="space-y-2">
                          {msg.citations.map((c) => (
                            <div key={`${msg.id}-${c.index}`}>
                              <div className="font-medium text-slate-700">
                                [{c.index}] {c.source}
                              </div>
                              <div className="mt-0.5 leading-6">{c.excerpt}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>
              ))}

              {submitting ? (
                <div className="flex justify-start">
                  <div className="rounded-3xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600 shadow-sm inline-flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    正在生成回答...
                  </div>
                </div>
              ) : null}
            </div>

            <div className="mt-4 rounded-3xl border border-slate-200 bg-white p-3 shadow-sm">
              <div className="flex items-end gap-3">
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
                  className="min-h-[84px] flex-1 resize-none rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm outline-none transition focus:border-slate-400 focus:bg-white"
                />
                <button
                  onClick={handleSend}
                  disabled={submitting || !query.trim()}
                  className="inline-flex h-[84px] items-center justify-center gap-2 rounded-2xl bg-slate-900 px-5 text-sm font-medium text-white shadow-sm transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {submitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
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