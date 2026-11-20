import { NextResponse } from "next/server";
import { backendConfigured, callBackend } from "@/lib/backend";
import { summarizeEvidence, verifyQuestion } from "@/lib/evidence";
import { allowRequest } from "@/lib/rate-limit";
import type { EventRecord } from "@/lib/types";

export const runtime = "nodejs";

export async function POST(request: Request) {
  if (!allowRequest(request)) return NextResponse.json({ error: "Rate limit exceeded." }, { status: 429 });
  try {
    const body = (await request.json()) as { question?: unknown; records?: unknown };
    if (typeof body.question !== "string" || body.question.length > 400) {
      return NextResponse.json({ error: "Question must be 1-400 characters." }, { status: 400 });
    }
    if (!Array.isArray(body.records) || body.records.length > 10_000) {
      return NextResponse.json({ error: "Provide at most 10,000 record objects." }, { status: 400 });
    }
    const records = body.records as EventRecord[];
    const evidence = summarizeEvidence(records);
    const verification = verifyQuestion(evidence, body.question);
    if (!backendConfigured()) {
      return NextResponse.json({
        ok: true,
        mode: "deterministic-degraded",
        content: "Verification is live. Scratch and Qwen inference are unavailable because no authenticated remote backend is configured.",
        evidence,
        scratchPrediction: "UNAVAILABLE",
        verifiedAnswer: verification.answer,
        verificationStatus: verification.status,
      });
    }
    const result = await callBackend("/v1/bridge", { question: body.question, records });
    return NextResponse.json({ ...result, evidence });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Bridge request failed.";
    return NextResponse.json({ ok: false, error: detail }, { status: 400 });
  }
}
