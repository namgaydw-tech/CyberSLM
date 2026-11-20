import { NextResponse } from "next/server";
import { backendConfigured } from "@/lib/backend";

export const runtime = "nodejs";

export async function GET() {
  return NextResponse.json({
    ok: true,
    deterministicEvidence: "available",
    remoteModels: backendConfigured() ? "configured" : "unavailable",
    identities: {
      scratch: "CyberSLM-35M (scratch-trained, structured evidence only)",
      conversational: "Qwen2.5-1.5B-Instruct (externally pretrained)",
    },
  });
}
