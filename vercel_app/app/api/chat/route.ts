import { NextResponse } from "next/server";
import { callBackend } from "@/lib/backend";
import { allowRequest } from "@/lib/rate-limit";

export const runtime = "nodejs";

export async function POST(request: Request) {
  if (!allowRequest(request)) return NextResponse.json({ error: "Rate limit exceeded." }, { status: 429 });
  try {
    const body = (await request.json()) as { message?: unknown; history?: unknown };
    if (typeof body.message !== "string" || !body.message.trim() || body.message.length > 2_000) {
      return NextResponse.json({ error: "Message must be 1-2,000 characters." }, { status: 400 });
    }
    const result = await callBackend("/v1/chat", {
      message: body.message,
      history: Array.isArray(body.history) ? body.history.slice(-12) : [],
    });
    return NextResponse.json(result);
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Remote backend failed.";
    return NextResponse.json(
      {
        ok: false,
        error: "Conversational inference is unavailable; no answer was fabricated.",
        detail,
      },
      { status: 503 },
    );
  }
}
