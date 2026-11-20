import type { EventRecord, EvidenceSummary } from "./types";

const EVENT_KEYS = ["event_type", "eventtype", "operation_type", "op_type", "operation"];
const SOURCE_KEYS = ["source_entity", "source", "src", "src_ip", "subject"];
const DESTINATION_KEYS = ["destination_entity", "destination", "dst", "dst_ip", "object"];

function valueFor(record: EventRecord, aliases: string[]): unknown {
  const entries = Object.entries(record);
  for (const alias of aliases) {
    const match = entries.find(([key]) => key.toLowerCase() === alias);
    if (match) return match[1];
  }
  return undefined;
}

export function summarizeEvidence(records: EventRecord[]): EvidenceSummary {
  if (records.length === 0) throw new Error("Evidence file contains no records.");
  if (records.length > 10_000) throw new Error("Evidence file exceeds the 10,000-row limit.");
  const eventTypeCounts: Record<string, number> = {};
  const sources = new Set<string>();
  const destinations = new Set<string>();
  for (const record of records) {
    const eventType = valueFor(record, EVENT_KEYS);
    if (eventType !== undefined && String(eventType).trim()) {
      const label = String(eventType).trim();
      eventTypeCounts[label] = (eventTypeCounts[label] ?? 0) + 1;
    }
    const source = valueFor(record, SOURCE_KEYS);
    if (source !== undefined && String(source).trim()) sources.add(String(source));
    const destination = valueFor(record, DESTINATION_KEYS);
    if (destination !== undefined && String(destination).trim()) {
      destinations.add(String(destination));
    }
  }
  return {
    rowsAvailable: records.length,
    eventTypeCounts,
    distinctSources: sources.size || null,
    distinctDestinations: destinations.size || null,
    verification:
      "Deterministic counts describe the uploaded records; they do not prove an attack.",
  };
}

export function verifyQuestion(summary: EvidenceSummary, question: string) {
  const normalized = question.trim().replace(/[?!.]+$/, "").toLowerCase();
  if (/^(count|how many|number of).*(unique|distinct).*event types/.test(normalized)) {
    return { status: "VERIFIED", answer: String(Object.keys(summary.eventTypeCounts).length) };
  }
  const eventMatch = normalized.match(/(?:count|how many).*?(event_[a-z0-9_]+)/);
  if (eventMatch) {
    const wanted = eventMatch[1].toUpperCase();
    const count = Object.entries(summary.eventTypeCounts).find(
      ([key]) => key.toUpperCase() === wanted,
    )?.[1] ?? 0;
    return { status: "VERIFIED", answer: String(count) };
  }
  if (/confirm.*attack|prove.*attack/.test(normalized)) {
    return { status: "NOT_ESTABLISHED", answer: "NO" };
  }
  return { status: "UNSUPPORTED_QUESTION", answer: null };
}
