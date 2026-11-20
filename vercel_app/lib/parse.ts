import { ParquetReader } from "parquetjs-lite";
import type { EventRecord } from "./types";

export const MAX_UPLOAD_BYTES = 1_000_000;

function parseCsvLine(line: string): string[] {
  const values: string[] = [];
  let current = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"' && quoted && line[index + 1] === '"') {
      current += '"';
      index += 1;
    } else if (character === '"') {
      quoted = !quoted;
    } else if (character === "," && !quoted) {
      values.push(current);
      current = "";
    } else {
      current += character;
    }
  }
  values.push(current);
  return values;
}

function parseCsv(text: string): EventRecord[] {
  const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) throw new Error("CSV requires a header and at least one data row.");
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) =>
    Object.fromEntries(headers.map((header, index) => [header, parseCsvLine(line)[index] ?? ""])),
  );
}

function validateRecords(value: unknown): EventRecord[] {
  const candidate =
    value && typeof value === "object" && !Array.isArray(value)
      ? Object.values(value as Record<string, unknown>).find(Array.isArray)
      : value;
  if (!Array.isArray(candidate) || candidate.some((item) => !item || typeof item !== "object")) {
    throw new Error("Expected a JSON array of record objects.");
  }
  return candidate as EventRecord[];
}

export async function parseEvidenceFile(file: File): Promise<EventRecord[]> {
  if (file.size > MAX_UPLOAD_BYTES) throw new Error("File exceeds the 1 MB deployment limit.");
  const extension = file.name.toLowerCase().split(".").pop();
  if (extension === "parquet" || extension === "pq") {
    const reader = await ParquetReader.openBuffer(Buffer.from(await file.arrayBuffer()));
    const cursor = reader.getCursor();
    const rows: EventRecord[] = [];
    try {
      while (rows.length <= 10_000) {
        const row = await cursor.next();
        if (!row) break;
        rows.push(row);
      }
    } finally {
      await reader.close();
    }
    return validateRecords(rows);
  }
  const text = await file.text();
  if (extension === "csv") return parseCsv(text);
  if (extension === "jsonl") {
    return validateRecords(
      text.split(/\r?\n/).filter((line) => line.trim()).map((line) => JSON.parse(line)),
    );
  }
  if (extension === "json") return validateRecords(JSON.parse(text));
  throw new Error("Supported formats: CSV, JSON, JSONL, and Parquet.");
}
