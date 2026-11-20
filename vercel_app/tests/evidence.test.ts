import { describe, expect, it } from "vitest";
import { summarizeEvidence, verifyQuestion } from "../lib/evidence";

describe("deterministic evidence", () => {
  it("counts exactly 256 records", () => {
    const records = Array.from({ length: 256 }, (_, index) => ({
      event_type: index % 2 ? "EVENT_READ" : "EVENT_WRITE",
    }));
    const result = summarizeEvidence(records);
    expect(result.rowsAvailable).toBe(256);
    expect(result.eventTypeCounts).toEqual({ EVENT_WRITE: 128, EVENT_READ: 128 });
  });

  it("does not turn counts into an attack verdict", () => {
    const summary = summarizeEvidence([{ event_type: "EVENT_READ" }]);
    expect(verifyQuestion(summary, "Do these counts confirm an attack?")).toEqual({
      status: "NOT_ESTABLISHED",
      answer: "NO",
    });
  });
});
