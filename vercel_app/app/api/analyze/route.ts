import { NextResponse } from "next/server";
import { summarizeEvidence } from "@/lib/evidence";
import { parseEvidenceFile } from "@/lib/parse";
import { allowRequest } from "@/lib/rate-limit";

export const runtime = "nodejs";

export async function POST(request: Request) {
  if (!allowRequest(request)) return NextResponse.json({ error: "Rate limit exceeded." }, { status: 429 });
  try {
    const form = await request.formData();
    const file = form.get("file");
    if (!(file instanceof File)) throw new Error("Attach one evidence file.");
    const records = await parseEvidenceFile(file);
    return NextResponse.json({ ok: true, evidence: summarizeEvidence(records), records });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Evidence parsing failed.";
    return NextResponse.json({ ok: false, error: message }, { status: 400 });
  }
}
