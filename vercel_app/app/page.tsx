"use client";

import { useEffect, useState } from "react";
import type { ApiResult, EventRecord, EvidenceSummary } from "@/lib/types";

type Mode = "chat" | "evidence" | "bridge";
type Message = { role: "user" | "assistant"; content: string };

const SAMPLE_RECORDS: EventRecord[] = [
  { event_type: "EVENT_READ", source_entity: "fictional-host-a", destination_entity: "file-1" },
  { event_type: "EVENT_WRITE", source_entity: "fictional-host-a", destination_entity: "file-2" },
  { event_type: "EVENT_READ", source_entity: "fictional-host-b", destination_entity: "file-1" },
];

export default function Home() {
  const [mode, setMode] = useState<Mode>("chat");
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [records, setRecords] = useState<EventRecord[]>(SAMPLE_RECORDS);
  const [evidence, setEvidence] = useState<EvidenceSummary | null>(null);
  const [status, setStatus] = useState("checking");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch("/api/status")
      .then((response) => response.json())
      .then((result) => setStatus(result.remoteModels))
      .catch(() => setStatus("unreachable"));
  }, []);

  const identity = mode === "chat"
    ? "Qwen2.5-1.5B-Instruct · externally pretrained"
    : mode === "bridge"
      ? "CyberSLM-35M → verification → Qwen"
      : "Deterministic local evidence verifier";

  async function upload(file: File) {
    setBusy(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch("/api/analyze", { method: "POST", body: form });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error);
      setEvidence(result.evidence);
      setRecords(result.records);
    } catch (error) {
      setMessages((items) => [...items, { role: "assistant", content: String(error) }]);
    } finally {
      setBusy(false);
    }
  }

  async function submit() {
    const current = message.trim();
    if (!current || busy) return;
    setMessage("");
    setMessages((items) => [...items, { role: "user", content: current }]);
    if (mode === "evidence") {
      setMessages((items) => [
        ...items,
        {
          role: "assistant",
          content: evidence
            ? `${evidence.rowsAvailable} rows parsed. Counts: ${JSON.stringify(evidence.eventTypeCounts)}. ${evidence.verification}`
            : "Upload a CSV, JSON, JSONL, or Parquet file to compute verified counts.",
        },
      ]);
      return;
    }
    setBusy(true);
    try {
      const endpoint = mode === "bridge" ? "/api/bridge" : "/api/chat";
      const body = mode === "bridge"
        ? { question: current, records: records.length ? records : SAMPLE_RECORDS }
        : { message: current, history: messages.slice(-12) };
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      const result = (await response.json()) as ApiResult & { error?: string; detail?: string };
      const content = result.content ?? `${result.error ?? "Request failed"}${result.detail ? ` (${result.detail})` : ""}`;
      setMessages((items) => [...items, { role: "assistant", content }]);
      if (result.evidence) setEvidence(result.evidence);
    } catch (error) {
      setMessages((items) => [
        ...items,
        { role: "assistant", content: `Request failed without a fabricated answer: ${String(error)}` },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <aside>
        <div className="brand"><span>CS</span><div><strong>CyberSLM</strong><small>RESEARCH LAB</small></div></div>
        <label htmlFor="mode">Capability</label>
        <select id="mode" value={mode} onChange={(event) => setMode(event.target.value as Mode)}>
          <option value="chat">Conversational assistant</option>
          <option value="evidence">Verified evidence analysis</option>
          <option value="bridge">Two-model bridge</option>
        </select>
        <div className="identity"><small>ACTIVE IDENTITY</small><p>{identity}</p></div>
        <div className={`health ${status === "configured" ? "online" : "degraded"}`}>
          <span /> Remote models: {status}
        </div>
        <label className="upload">
          Evidence file
          <input type="file" accept=".csv,.json,.jsonl,.parquet,.pq" onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void upload(file);
          }} />
          <small>CSV, JSON, JSONL, Parquet · 1 MB max</small>
        </label>
        <button className="secondary" onClick={() => { setMessages([]); setEvidence(null); }}>Reset session</button>
      </aside>
      <section className="workspace">
        <header>
          <div><small>FAULT-TOLERANT SECURITY ASSISTANT</small><h1>CyberSLM Console</h1></div>
          <a href="https://github.com/namgaydw-tech/CyberSLM">Source repository ↗</a>
        </header>
        <div className="notice">
          Deterministic evidence verification is live on Vercel. Model answers require an authenticated remote backend; unavailable services never produce simulated answers.
        </div>
        {evidence && (
          <div className="evidence">
            <strong>Verified evidence</strong>
            <span>{evidence.rowsAvailable} records</span>
            <code>{JSON.stringify(evidence.eventTypeCounts)}</code>
          </div>
        )}
        <div className="messages" aria-live="polite">
          {messages.length === 0 && (
            <div className="empty"><div>⌁</div><h2>Ask a defensive cybersecurity question</h2><p>Or upload fictional/authorized event evidence for deterministic analysis.</p></div>
          )}
          {messages.map((item, index) => (
            <article className={item.role} key={`${item.role}-${index}`}>
              <small>{item.role === "user" ? "YOU" : "CYBERSLM"}</small>
              <p>{item.content}</p>
            </article>
          ))}
        </div>
        <div className="composer">
          <label className="sr-only" htmlFor="question">Question</label>
          <textarea
            id="question"
            value={message}
            maxLength={2_000}
            placeholder={mode === "bridge" ? "Try: count unique event types" : "Type your question…"}
            onChange={(event) => setMessage(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void submit(); }
            }}
          />
          <button disabled={busy || !message.trim()} onClick={() => void submit()}>{busy ? "Working…" : "Send"}</button>
        </div>
        <footer>CyberSLM-35M is scratch-trained. Qwen2.5-1.5B-Instruct is externally pretrained. Outputs require human review.</footer>
      </section>
    </main>
  );
}
